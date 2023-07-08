use anyhow::{anyhow, Result};
use derive_more::{Deref, IsVariant, Unwrap};
use itertools::Itertools;
use std::collections::HashSet;
use std::ops::ControlFlow;
use std::sync::atomic::AtomicUsize;
use std::sync::Arc;
use std::{collections::HashMap, rc::Rc};

use super::{
    BindingDescription, Buffer, Resource, ResourceDescription, SampledImage, UniformBuffer,
};
use crate::collections::{ParitySet, Redundancy, RedundantSet};
use crate::{collections::FrequencySet, material::ShaderModule, prelude::*};

/// ### Warning
/// descriptor set layouts held inside a layout are not destroyed on
/// [`Destructible::destroy`], and instead are owned by the
/// [`Layouts`] struct, and destroyed with it
#[derive(Debug, Clone)]
pub struct Layout {
    pub pipeline: vk::PipelineLayout,
    pub descriptors: FrequencySet<Option<vk::DescriptorSetLayout>>,
}

impl Destructible for Layout {
    fn destroy(self, loader: &Loader) {
        self.pipeline.destroy(loader);
    }
}

#[derive(Debug, Clone, Deref)]
pub struct Layouts {
    descriptors_flat: Vec<vk::DescriptorSetLayout>,
    #[deref]
    pub layouts: HashMap<Identifier, Layout>,
}

impl Destructible for Layouts {
    fn destroy(self, loader: &Loader) {
        self.descriptors_flat.destroy(loader);
        self.layouts.into_values().destroy(loader);
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum Discriminant {
    Global,
    Local(Identifier, vk::DescriptorFrequency),
}

fn equivalient_binding(
    lhs: vk::DescriptorSetLayoutBinding,
    rhs: vk::DescriptorSetLayoutBinding,
) -> bool {
    lhs.binding == rhs.binding
        && lhs.descriptor_count == rhs.descriptor_count
        && lhs.descriptor_type == rhs.descriptor_type
        && lhs.p_immutable_samplers == rhs.p_immutable_samplers
}

fn consolidate_bindings<I: IntoIterator<Item = vk::DescriptorSetLayoutBinding>>(
    iter: I,
) -> Result<Vec<vk::DescriptorSetLayoutBinding>> {
    iter.into_iter()
        .sorted_by_key(|binding| binding.binding)
        .try_fold(Vec::new(), |mut acc, binding| {
            if let Some(back) = acc.last().copied() {
                if equivalient_binding(back, binding) {
                    let back = vk::DescriptorSetLayoutBinding {
                        stage_flags: back.stage_flags | binding.stage_flags,
                        ..binding
                    };

                    *acc.last_mut().unwrap() = back;
                } else if back.binding == binding.binding {
                    return Err(anyhow!(
                        "Bindings match, but other elements not equivalent: {:?}, {:?}",
                        back,
                        binding
                    ));
                } else {
                    acc.push(binding)
                }
            } else {
                acc.push(binding)
            }

            Ok(acc)
        })
}

pub fn build_layout<'a, S: 'a>(loader: &Loader, shaders: S) -> Result<Layouts>
where
    S: IntoIterator<Item = (Identifier, &'a ShaderModule)> + Clone,
{
    let descriptor_layouts = shaders
        .clone()
        .into_iter()
        .flat_map(|(id, module)| {
            let stage_flags = module.stage_flags;

            module
                .resources
                .iter()
                .filter_map(|resource| resource.get_shader_binding())
                .map(move |desc| {
                    let binding = vk::DescriptorSetLayoutBinding {
                        binding: desc.binding,
                        descriptor_type: desc.ty,
                        descriptor_count: desc.count,
                        stage_flags,
                        ..Default::default()
                    };

                    let discriminant = if desc.frequency == vk::DescriptorFrequency::Global {
                        Discriminant::Global
                    } else {
                        Discriminant::Local(id.clone(), desc.frequency)
                    };

                    (discriminant, binding)
                })
        })
        .into_group_map()
        .into_iter()
        .map(|(discriminant, bindings)| {
            let bindings = consolidate_bindings(bindings)?;

            let create_info = vk::DescriptorSetLayoutCreateInfo::builder().bindings(&bindings);

            let layout = unsafe {
                loader
                    .device
                    .create_descriptor_set_layout(&create_info, None)?
            };

            Ok((discriminant, layout))
        })
        .collect::<Result<HashMap<_, _>>>()?;

    let layouts = shaders
        .into_iter()
        .map(|(id, _)| id)
        .collect::<HashSet<_>>()
        .into_iter()
        .map(|id| {
            let descriptors = vk::DescriptorFrequency::ELEMENTS
                .into_iter()
                .filter_map(|freq| {
                    let layout = if freq == vk::DescriptorFrequency::Global {
                        descriptor_layouts.get(&Discriminant::Global).cloned()
                    } else {
                        let discriminant = Discriminant::Local(id.clone(), freq);
                        descriptor_layouts.get(&discriminant).cloned()
                    };

                    layout.map(|layout| (freq, layout))
                })
                .collect::<FrequencySet<Vec<_>>>()
                .map(|layouts| layouts.get(0).copied());
            
            let flattened = descriptors.values().copied().flatten().collect_vec();
            let create_info = vk::PipelineLayoutCreateInfo::builder().set_layouts(&flattened);

            let pipeline_layout =
                unsafe { loader.device.create_pipeline_layout(&create_info, None) };

            pipeline_layout.map(|pipeline| {
                (
                    id.clone(),
                    Layout {
                        pipeline,
                        descriptors,
                    }
                )
            })
        })
        .collect::<std::result::Result<HashMap<_, _>, _>>()?;

    let descriptors_flat = descriptor_layouts.into_values().collect::<Vec<_>>();

    Ok(Layouts {
        descriptors_flat,
        layouts,
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
