use crate::{material::ShaderModule, pipeline::Shader, prelude::*, resources::ParitySet};

use anyhow::{anyhow, Result};
use impl_trait_for_tuples::impl_for_tuples;
use itertools::{izip, Itertools};
use std::collections::HashMap;

pub trait BindableVec {
    fn bindings(&self) -> Vec<BindingDescription>;
}

pub trait Bindable: Copy + Default {
    fn binding(&self) -> BindingDescription;
    fn pool_size(&self) -> vk::DescriptorPoolSize {
        vk::DescriptorPoolSize::builder()
            .descriptor_count(self.binding().descriptor_count)
            .ty(self.binding().ty)
            .build()
    }
}

#[derive(Clone, Copy, Debug)]
pub enum ResourceType {
    Image,
    Buffer,
}

pub trait DescriptorWriter {
    fn get_write<'a>(&'a self) -> DescriptorWrite<'a>;
}

pub struct DescriptorWrite<'a> {
    pub resource_ty: ResourceType,
    pub buffer_info: ParitySet<vk::DescriptorBufferInfoBuilder<'a>>,
    pub image_info: ParitySet<vk::DescriptorImageInfoBuilder<'a>>,
    pub binding: BindingDescription,
}

impl<'a> DescriptorWrite<'a> {
    pub fn from_buffer(
        buffer_info: ParitySet<vk::DescriptorBufferInfoBuilder<'a>>,
        binding: BindingDescription,
    ) -> Self {
        Self {
            resource_ty: ResourceType::Buffer,
            buffer_info,
            image_info: ParitySet::from_fn(|| vk::DescriptorImageInfo::builder()),
            binding,
        }
    }

    pub fn from_image(
        image_info: ParitySet<vk::DescriptorImageInfoBuilder<'a>>,
        binding: BindingDescription,
    ) -> Self {
        Self {
            resource_ty: ResourceType::Image,
            buffer_info: ParitySet::from_fn(|| vk::DescriptorBufferInfo::builder()),
            image_info,
            binding,
        }
    }

    fn write(&self, loader: &Loader, sets: &ParitySet<vk::DescriptorSet>) {
        unsafe {
            loader.device.update_descriptor_sets(
                &izip!(self.buffer_info.iter(), self.image_info.iter(), sets.iter())
                    .map(|(buffer_info, image_info, set)| {
                        let mut write = vk::WriteDescriptorSet::builder()
                            .descriptor_type(self.binding.ty)
                            .dst_binding(self.binding.binding)
                            .dst_set(*set);

                        match self.resource_ty {
                            ResourceType::Image => {
                                write = write.image_info(std::slice::from_ref(image_info))
                            }
                            ResourceType::Buffer => {
                                write = write.buffer_info(std::slice::from_ref(buffer_info))
                            }
                        }

                        write.build()
                    })
                    .collect_vec(),
                &[],
            );
        }
    }
}

impl<T: Bindable> BindableVec for T {
    fn bindings(&self) -> Vec<BindingDescription> {
        vec![self.binding()]
    }
}

#[impl_for_tuples(1, 10)]
impl BindableVec for Tuple {
    fn bindings(&self) -> Vec<BindingDescription> {
        [for_tuples!( #( Tuple.bindings() ),* )]
            .into_iter()
            .flat_map(|v| v.into_iter())
            .collect()
    }
}

impl BindableVec for () {
    fn bindings(&self) -> Vec<BindingDescription> {
        vec![]
    }
}

#[derive(Debug, Clone)]
pub struct Layouts {
    pub descriptors: HashMap<DescriptorFrequency, vk::DescriptorSetLayout>,
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

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum DescriptorFrequency {
    #[default]
    Global,
    Pass,
    Material,
    Object,
}

#[derive(PartialEq, Clone, Copy, Debug)]
pub struct BindingDescription {
    pub ty: vk::DescriptorType,
    pub frequency: DescriptorFrequency,
    pub binding: u32,
    pub descriptor_count: u32,
    pub stage_flags: vk::ShaderStageFlags,
}

impl Default for BindingDescription {
    fn default() -> Self {
        Self {
            ty: vk::DescriptorType::default(),
            frequency: DescriptorFrequency::Global,
            binding: 0,
            descriptor_count: 1,
            stage_flags: vk::ShaderStageFlags::empty(),
        }
    }
}

pub fn get_layouts(loader: &Loader, shaders: &[&dyn Shader]) -> Result<Layouts> {
    let descriptors = shaders
        .iter()
        .flat_map(|s| s.descriptor_bindings())
        .into_grouping_map_by(|binding| binding.frequency)
        .fold(
            HashMap::<u32, BindingDescription>::new(),
            |mut acc, _, binding| {
                if let Some(old) = acc.get_mut(&binding.binding) {
                    assert!(
                        old.ty == binding.ty && old.descriptor_count == binding.descriptor_count
                    );
                    old.stage_flags |= binding.stage_flags;
                } else {
                    acc.insert(binding.binding, binding);
                };

                acc
            },
        )
        .into_iter()
        .map(|(k, v)| {
            let bindings = v
                .into_values()
                .map(|binding| {
                    vk::DescriptorSetLayoutBinding::builder()
                        .descriptor_count(binding.descriptor_count)
                        .stage_flags(binding.stage_flags)
                        .descriptor_type(binding.ty)
                        .binding(binding.binding)
                        .build()
                })
                .collect_vec();

            let ci = vk::DescriptorSetLayoutCreateInfo::builder().bindings(&bindings);
            let descriptor = unsafe {
                loader
                    .device
                    .create_descriptor_set_layout(&ci, None)
                    .unwrap()
            };

            (k, descriptor)
        })
        .collect::<HashMap<_, _>>();

    let set_layouts = descriptors.values().map(|&v| v).collect_vec();
    let pipeline_ci = vk::PipelineLayoutCreateInfo::builder().set_layouts(&set_layouts);
    let pipeline = unsafe { loader.device.create_pipeline_layout(&pipeline_ci, None)? };

    Ok(Layouts {
        descriptors,
        pipeline,
    })
}

pub unsafe fn get_descriptors<'a>(
    loader: &Loader,
    layouts: &Layouts,
    writes: impl IntoIterator<Item = &'a DescriptorWrite<'a>> + Clone,
) -> Result<(
    vk::DescriptorPool,
    HashMap<DescriptorFrequency, ParitySet<vk::DescriptorSet>>,
)> {
    let pool_sizes = writes
        .clone()
        .into_iter()
        .map(|write| {
            let binding = write.binding;
            vk::DescriptorPoolSize {
                ty: binding.ty,
                descriptor_count: binding.descriptor_count * 2,
            }
        })
        .collect_vec();

    let pool_ci = vk::DescriptorPoolCreateInfo::builder()
        .pool_sizes(&pool_sizes)
        .max_sets(8);

    let pool = loader.device.create_descriptor_pool(&pool_ci, None)?;

    let sets = layouts
        .descriptors
        .iter()
        .map(|(&freq, &layout)| {
            let layouts = [layout; 2];
            let alloc_info = vk::DescriptorSetAllocateInfo::builder()
                .descriptor_pool(pool)
                .set_layouts(&layouts);

            let sets = ParitySet::from(loader.device.allocate_descriptor_sets(&alloc_info)?);

            Ok((freq, sets))
        })
        .collect::<Result<HashMap<_, _>>>()?;

    for write in writes {
        let set = sets.get(&write.binding.frequency).unwrap();
        write.write(loader, &set);
    }

    Ok((pool, sets))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct PartialShaderBinding {
    pub frequency: DescriptorFrequency,
    pub binding: u32,
    pub count: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct ShaderBinding {
    pub ty: vk::DescriptorType,
    pub frequency: DescriptorFrequency,
    pub binding: u32,
    pub count: u32,
}

impl PartialShaderBinding {
    pub fn as_binding(self) -> ShaderBinding {
        ShaderBinding {
            ty: Default::default(),
            frequency: self.frequency,
            binding: self.binding,
            count: self.count,
        }
    }
}

pub trait VertexInput {
    fn bindings() -> Vec<vk::VertexInputBindingDescription>;
    fn attributes() -> Vec<vk::VertexInputAttributeDescription>;
}
