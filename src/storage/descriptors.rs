use crate::{loader::Loader, pipeline::Shader, prelude::*};
use anyhow::Result;
use impl_trait_for_tuples::impl_for_tuples;
use itertools::Itertools;

pub trait BindableVec {
    fn bindings(&self) -> Vec<vk::DescriptorSetLayoutBindingBuilder>;
}

pub trait Bindable {
    fn binding(&self) -> vk::DescriptorSetLayoutBindingBuilder;
}

impl<T: Bindable> BindableVec for T {
    fn bindings(&self) -> Vec<vk::DescriptorSetLayoutBindingBuilder> {
        vec![self.binding()]
    }
}

#[impl_for_tuples(1, 10)]
impl BindableVec for Tuple {
    fn bindings(&self) -> Vec<vk::DescriptorSetLayoutBindingBuilder> {
        [for_tuples!( #( Tuple.bindings() ),* )]
            .into_iter()
            .flat_map(|v| v.into_iter())
            .collect()
    }
}

impl BindableVec for () {
    fn bindings(&self) -> Vec<vk::DescriptorSetLayoutBindingBuilder> {
        vec![]
    }
}

pub struct Layouts {
    pub descriptor: vk::DescriptorSetLayout,
    pub pipeline: vk::PipelineLayout,
}

pub fn get_layouts(loader: &Loader, shaders: &[&dyn Shader]) -> Result<Layouts> {
    let bindings = shaders
        .iter()
        .flat_map(|s| s.descriptor_bindings())
        .collect_vec();

    let descriptor_ci = vk::DescriptorSetLayoutCreateInfo::builder().bindings(&bindings);
    let descriptor = unsafe {
        loader
            .device
            .create_descriptor_set_layout(&descriptor_ci, None)?
    };

    let pipeline_ci =
        vk::PipelineLayoutCreateInfo::builder().set_layouts(std::slice::from_ref(&descriptor));
    let pipeline = unsafe { loader.device.create_pipeline_layout(&pipeline_ci, None)? };

    Ok(Layouts {
        descriptor,
        pipeline,
    })
}
