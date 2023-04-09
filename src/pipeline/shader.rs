use crate::prelude::*;
use anyhow::{Result};

pub struct VertexShaderCreateInfo {

}

pub struct VertexShader {
    pub shader: vk::ShaderModule,
    pub entry: &'static str,
    bindings: Vec<Box<dyn Bindable>>,
    vertex: Box<dyn Vertex>
}

impl VertexShader {
    fn new(create_info: VertexShaderCreateInfo) -> Result<Self> {
        
    }
}

pub trait Vertex {
    fn bindings(&self) -> Vec<vk::VertexInputBindingDescription>;
    fn attributes(&self) -> Vec<vk::VertexInputBindingDescription>;
}

pub trait Bindable {
    fn binding(&self) -> vk::DescriptorSetLayoutBinding;
}