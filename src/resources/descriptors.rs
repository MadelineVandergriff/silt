use anyhow::{anyhow, Result};
use derive_more::{IsVariant, Unwrap};
use itertools::Itertools;
use std::ops::ControlFlow;
use std::sync::Arc;
use std::sync::atomic::AtomicUsize;
use std::{collections::HashMap, rc::Rc};

use super::{
    BindingDescription, Buffer, Resource,
    ResourceDescription, SampledImage, UniformBuffer,
};
use crate::collections::{ParitySet, Redundancy, RedundantSet};
use crate::vk::DescriptorFrequency;
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

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum Discriminant {
    Global,
    Local(Identifier, vk::DescriptorFrequency)
}

pub fn build_layout<'a, S: 'a>(loader: &Loader, shaders: S) -> Result<()>
where
    S: IntoIterator<Item = (Identifier, &'a ShaderModule)>,
{
    let descriptors = shaders
        .into_iter()
        .flat_map(|(id, shader)| {
            shader.resources
                .iter()
                .cloned()
                .map(move |resource| (id.clone(), shader.stage_flags, resource))
        })
        .filter_map(|(id, stage, resource)| {
            resource
                .get_shader_binding()
                .map(|binding| (id, stage, binding))
        })
        .group_by(|(id, _, binding)| (id.clone(), binding.frequency, binding.binding))
        .into_iter()
        .map(|(_, group)| {
            let mut peekable = group.peekable();
            let init = peekable.peek().unwrap().clone();

            peekable
                .try_fold(init, |acc, (_, flags, binding)| {
                    if acc.2 == binding {
                        Ok((acc.0, acc.1 | flags, binding))
                    } else {
                        Err(anyhow!("failed to accumulate bindings"))
                    }
                })
        })
        .collect::<Result<Vec<(Identifier, vk::ShaderStageFlags, BindingDescription)>>>()?
        .into_iter()
        .group_by(|(id, _, binding)| {
            if binding.frequency == vk::DescriptorFrequency::Global {
                Discriminant::Global
            } else {
                Discriminant::Local(id.clone(), binding.frequency)
            }
        })
        .into_iter()
        .map(|(discriminant, group)| {
            let bindings = group
                .map(|(id, flags, binding)| {
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
            let (frequency, id) = match discriminant {
                Discriminant::Global => (vk::DescriptorFrequency::Global, None),
                Discriminant::Local(id, freq) => (freq, Some(id)),
            };
            layout.map(|layout| (frequency, (layout, id)))
        })
        .collect::<std::result::Result<FrequencySet<Vec<_>>, _>>()?;

    if descriptors.global.len() != 1 {
        return Err(anyhow!("Not exactly one global descriptor set layout"))
    }

    let global = descriptors.global.first().unwrap().0;

    let pipeline_layouts = descriptors.iter()
        .filter(|(freq, _)| *freq != vk::DescriptorFrequency::Global)
        .flat_map(|(_, layout)| layout.iter())
        .map(|(layout, id)| {
            
        })

    /*let set_layouts_flat = descriptors.values().map(|&l| l).collect_vec();
    let pipeline_layout_ci = vk::PipelineLayoutCreateInfo::builder().set_layouts(&set_layouts_flat);

    let pipeline = unsafe {
        loader
            .device
            .create_pipeline_layout(&pipeline_layout_ci, None)?
    };

    Ok(Layouts {
        descriptors,
        pipeline,
    })*/

    Ok(())
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
        sets: FrequencySet<ParitySet<vk::DescriptorSet>>,
    ) -> Result<()> {
        if let Some(binding) = self.description.get_shader_binding() {
            let references = self
                .reference
                .as_type(Redundancy::Parity, None)
                .map_err(|_| anyhow!("Resources corresponding to a descriptor set must be either single or a parity set"))?
                .unwrap_parity();

            for (reference, set) in std::iter::zip(references, sets.get(binding.frequency)) {
                reference.write_descriptor(&binding, loader, *set);
            }
        }

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
