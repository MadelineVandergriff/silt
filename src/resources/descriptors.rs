use anyhow::{anyhow, Result};
use derive_more::{IsVariant, Unwrap};
use itertools::Itertools;
use std::{collections::HashMap, rc::Rc};

use super::{
    BindingDescription, Buffer, Resource,
    ResourceDescription, SampledImage, UniformBuffer,
};
use crate::collections::{ParitySet, Redundancy, RedundantSet};
use crate::{material::ShaderModule, prelude::*, collections::FrequencySet};



#[derive(Debug, Clone)]
pub struct Layouts {
    pub descriptors: FrequencySet<vk::DescriptorSetLayout>,
    pub pipeline: vk::PipelineLayout,
}

impl Destructible for Layouts {
    fn destroy(self, loader: &Loader) {
        for descriptor_layout in self.descriptors.into_values() {
            descriptor_layout.destroy(loader);
        }
        self.pipeline.destroy(loader);
    }
}

pub fn build_layout<'a, S: 'a>(loader: &Loader, shaders: S) -> Result<Layouts>
where
    S: IntoIterator<Item = &'a ShaderModule>,
{
    let descriptors = shaders
        .into_iter()
        .flat_map(|shader| {
            std::iter::zip(
                [shader.stage_flags].into_iter().cycle(),
                shader.resources.clone(),
            )
        })
        .filter_map(|(stage, resource)| {
            resource
                .get_shader_binding()
                .map(|binding| (stage, binding))
        })
        .group_by(|(_, binding)| (binding.binding, binding.frequency))
        .into_iter()
        .map(|(_, group)| {
            let reduced = group
                .reduce(|(acc_flags, acc_binding), (flags, binding)| {
                    if acc_binding == binding {
                        (acc_flags | flags, acc_binding)
                    } else {
                        Default::default()
                    }
                })
                .unwrap();

            if reduced == Default::default() {
                Err(anyhow!("Failed to accumulate bindings"))
            } else {
                Ok(reduced)
            }
        })
        .collect::<Result<Vec<_>>>()?
        .into_iter()
        .group_by(|(_, binding)| binding.frequency)
        .into_iter()
        .map(|(frequency, group)| {
            let bindings = group
                .map(|(flags, binding)| {
                    vk::DescriptorSetLayoutBinding::builder()
                        .binding(binding.binding)
                        .descriptor_count(binding.count)
                        .descriptor_type(binding.ty)
                        .stage_flags(flags)
                        .build()
                })
                .collect_vec();

            let layout_ci = vk::DescriptorSetLayoutCreateInfo::builder().bindings(&bindings);
            let layout = unsafe { loader.device.create_descriptor_set_layout(&layout_ci, None) };
            layout.map(|layout| (frequency, layout))
        })
        .collect::<std::result::Result<FrequencySet<Vec<_>>, _>>()?
        .flatten()
        .ok_or(anyhow!("Multiple descriptor layouts per frequency"))?;

    let set_layouts_flat = descriptors.values().map(|&l| l).collect_vec();
    let pipeline_layout_ci = vk::PipelineLayoutCreateInfo::builder().set_layouts(&set_layouts_flat);

    let pipeline = unsafe {
        loader
            .device
            .create_pipeline_layout(&pipeline_layout_ci, None)?
    };

    Ok(Layouts {
        descriptors,
        pipeline,
    })
}

#[derive(Debug, Clone, Copy, IsVariant, Unwrap)]
pub enum ResourceReference<'a> {
    Buffer(&'a Buffer),
    Image(&'a SampledImage),
}

impl ResourceReference<'_> {
    pub fn write_descriptor(
        &self,
        binding: &BindingDescription,
        loader: &Loader,
        set: vk::DescriptorSet,
    ) {
        let buffer_info = match self {
            Self::Buffer(buffer) => Some(
                vk::DescriptorBufferInfo::builder()
                    .buffer(buffer.buffer)
                    .range(buffer.size)
                    .offset(0),
            ),
            _ => None,
        };

        let image_info = match self {
            Self::Image(image) => {
                Some(
                    vk::DescriptorImageInfo::builder()
                        .image_view(image.image.view)
                        .sampler(image.sampler)
                        .image_layout(image.image.layout.get().get_layout()), // TODO this could easily be wrong??
                )
            }
            _ => None,
        };

        let mut descriptor_write = vk::WriteDescriptorSet::builder()
            .descriptor_type(binding.ty)
            .dst_binding(binding.binding)
            .dst_set(set);

        if let Some(buffer_info) = &buffer_info {
            descriptor_write = descriptor_write.buffer_info(std::slice::from_ref(buffer_info));
        }

        if let Some(image_info) = &image_info {
            descriptor_write = descriptor_write.image_info(std::slice::from_ref(image_info));
        }

        unsafe {
            loader
                .device
                .update_descriptor_sets(std::slice::from_ref(&descriptor_write), &[]);
        }
    }
}

pub struct ResourceBinding<'a> {
    pub description: Rc<ResourceDescription>,
    pub reference: RedundantSet<ResourceReference<'a>>,
}

impl ResourceBinding<'_> {
    pub fn write_descriptor_sets(
        &self,
        loader: &Loader,
        sets: ParitySet<vk::DescriptorSet>,
    ) -> Result<()> {
        if let Some(binding) = self.description.get_shader_binding() {
            let references = self
                .reference
                .as_type(Redundancy::Parity, None)
                .map_err(|_| anyhow!("Resources corresponding to a descriptor set must be either single or a parity set"))?
                .unwrap_parity();

            for (reference, set) in std::iter::zip(references, sets) {
                reference.write_descriptor(&binding, loader, set);
            }
        }

        todo!();
        Ok(())
    }
}

pub trait BindableResource {
    fn bind(&self) -> ResourceBinding;
}

impl<T: Copy> BindableResource for Resource<UniformBuffer<T>> {
    fn bind(&self) -> ResourceBinding {
        let description = self.description.clone();
        let reference = self
            .resource
            .get_buffers()
            .map(|&buffer| ResourceReference::Buffer(buffer))
            .into();

        ResourceBinding {
            description,
            reference,
        }
    }
}

impl BindableResource for Resource<SampledImage> {
    fn bind(&self) -> ResourceBinding {
        let description = self.description.clone();
        let reference = ResourceReference::Image(&self.resource).into();

        ResourceBinding {
            description,
            reference,
        }
    }
}
