use std::{collections::HashMap};

use crate::{loader::Loader, pipeline::Shader, prelude::*};
use anyhow::Result;
use impl_trait_for_tuples::impl_for_tuples;
use itertools::Itertools;
use ouroboros::self_referencing;

use super::buffer::{Buffer, FromTypedBuffer};

pub trait BindableVec {
    fn bindings(&self) -> Vec<BindingDescription>;
}

pub trait Bindable: Copy + Default {
    type Write: DescriptorWrite;

    fn binding(&self) -> BindingDescription;
    fn pool_size(&self) -> vk::DescriptorPoolSize;
}

pub trait DescriptorWrite {
    fn write(&mut self, loader: &Loader, sets: &ParitySet<vk::DescriptorSet>);
    fn binding(&self) -> BindingDescription;
    fn is_constant(&self) -> bool {
        false
    } 
}

pub trait DescriptorWriter {
    fn writer(&self) -> Box<dyn DescriptorWrite>;
}

#[self_referencing]
pub struct UniformWrite {
    binding: BindingDescription,
    info: vk::DescriptorBufferInfo,
    #[borrows(info)]
    #[covariant]
    write: vk::WriteDescriptorSetBuilder<'this>,
}

impl UniformWrite {
    pub fn create<T: Bindable + Default>(info: vk::DescriptorBufferInfo) -> UniformWrite {
        let binding = T::default().binding();

        UniformWriteBuilder {
            binding,
            info,
            write_builder: |info: &vk::DescriptorBufferInfo| {
                vk::WriteDescriptorSet::builder()
                    .buffer_info(std::slice::from_ref(info))
                    .descriptor_type(binding.descriptor_type)
                    .dst_binding(binding.binding)
                    .dst_array_element(0)
            },
        }
        .build()
    }
}

impl<T: Bindable + Copy> FromTypedBuffer<T> for UniformWrite {
    fn from(value: &Buffer) -> Self {
        let info = vk::DescriptorBufferInfo::builder()
            .buffer(value.buffer)
            .offset(0)
            .range(value.size)
            .build();

        Self::create::<T>(info)
    }
}

impl DescriptorWrite for UniformWrite {
    fn write(&mut self, loader: &Loader, sets: &ParitySet<vk::DescriptorSet>) {
        let write = self.with_write_mut(|write| {
            let mut dummy = vk::WriteDescriptorSet::builder();
            std::mem::swap(&mut dummy, write);
            dummy
        });

        unsafe {
            loader
                .device
                .update_descriptor_sets(&[write.dst_set(sets.even).build()], &[])
        }
    }

    fn binding(&self) -> BindingDescription {
        *self.borrow_binding()
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

pub struct Layouts {
    pub descriptors: HashMap<DescriptorFrequency, vk::DescriptorSetLayout>,
    pub pipeline: vk::PipelineLayout,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DescriptorFrequency {
    Global,
    Pass,
    Material,
    Object,
}

#[derive(PartialEq, Clone, Copy, Debug)]
pub struct BindingDescription {
    pub descriptor_type: vk::DescriptorType,
    pub frequency: DescriptorFrequency,
    pub binding: u32,
    pub descriptor_count: u32,
    pub stage_flags: vk::ShaderStageFlags,
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
                        old.descriptor_type == binding.descriptor_type
                            && old.descriptor_count == binding.descriptor_count
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
            let bindings = 
                v.into_values()
                    .map(|binding| {
                        vk::DescriptorSetLayoutBinding::builder()
                            .descriptor_count(binding.descriptor_count)
                            .stage_flags(binding.stage_flags)
                            .descriptor_type(binding.descriptor_type)
                            .binding(binding.binding)
                            .build()
                    })
                    .collect_vec();
            
            let ci = vk::DescriptorSetLayoutCreateInfo::builder()
                    .bindings(&bindings);
            let descriptor = unsafe { loader.device.create_descriptor_set_layout(&ci, None).unwrap() };

            (k, descriptor)
        })
        .collect::<HashMap<_, _>>();
    
    let set_layouts = descriptors.values().map(|&v| v).collect_vec();
    let pipeline_ci =
        vk::PipelineLayoutCreateInfo::builder().set_layouts(&set_layouts);
    let pipeline = unsafe { loader.device.create_pipeline_layout(&pipeline_ci, None)? };

    Ok(Layouts {
        descriptors,
        pipeline,
    })
}

pub unsafe fn get_descriptors(loader: &Loader, layouts: &Layouts, writes: Vec<Box<dyn DescriptorWrite>>) -> Result<(vk::DescriptorPool, HashMap<DescriptorFrequency, ParitySet<vk::DescriptorSet>>)> {
    let pool_sizes = writes.iter()
        .map(|write| {
            let binding = write.binding();
            vk::DescriptorPoolSize {
                ty: binding.descriptor_type,
                descriptor_count: binding.descriptor_count,
            }
        })
        .collect_vec();

    let pool_ci = vk::DescriptorPoolCreateInfo::builder()
        .pool_sizes(&pool_sizes)
        .max_sets(8);

    let pool = loader.device.create_descriptor_pool(&pool_ci, None)?;

    let sets = layouts.descriptors.iter()
        .map(|(&freq, &layout)| {
            let layouts = [layout; 2];
            let alloc_info = vk::DescriptorSetAllocateInfo::builder()
                .descriptor_pool(pool)
                .set_layouts(&layouts);

            let sets = ParitySet::from(loader.device.allocate_descriptor_sets(&alloc_info)?);

            Ok((freq, sets))
        })
        .collect::<Result<HashMap<_, _>>>()?;

    for mut write in writes {
        let set = sets.get(&write.binding().frequency).unwrap();
        write.as_mut().write(loader, &set);
    }

    Ok((pool, sets))
}
