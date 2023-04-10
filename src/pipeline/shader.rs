pub use crate::macros::ShaderCode;

use crate::{prelude::*, loader::Loader};
use std::rc::Rc;
use anyhow::Result;
use impl_trait_for_tuples::impl_for_tuples;

pub struct VertexShaderCreateInfo {
    pub code: ShaderCode,
    pub entry: &'static str
}

pub struct VertexShader {
    pub shader: vk::ShaderModule,
    pub entry: &'static str,
    bindings: Rc<dyn BindableVec>,
    vertex: Rc<dyn Vertex>,
}

impl VertexShader {
    pub fn new<V, B>(loader: &Loader, create_info: VertexShaderCreateInfo) -> Result<Self>
    where
        V: Vertex + Default + 'static,
        B: BindableVec + Default + 'static,
    {
        let shader_ci = vk::ShaderModuleCreateInfo::builder()
            .code(&create_info.code[..]);

        Ok(Self {
            shader: unsafe { loader.device.create_shader_module(&shader_ci, None)? },
            entry: create_info.entry,
            bindings: Rc::new(B::default()),
            vertex: Rc::new(V::default()),
        })
    }

    pub fn vertex_bindings(&self) -> Vec<vk::VertexInputBindingDescription> {
        self.vertex.bindings()
    }

    pub fn vertex_attributes(&self) -> Vec<vk::VertexInputAttributeDescription> {
        self.vertex.attributes()
    }

    pub fn descriptor_bindings(&self) -> Vec<vk::DescriptorSetLayoutBinding> {
        self.bindings.bindings()
    }
}

impl Shader for VertexShader {
    fn shader_flags() -> vk::ShaderStageFlags {
        vk::ShaderStageFlags::VERTEX
    }
}

pub struct FragmentShader {
    pub shader: vk::ShaderModule,
    pub entry: &'static str,
    bindings: Rc<dyn BindableVec>,
}

impl FragmentShader {
    pub fn new<B>(loader: &Loader, create_info: VertexShaderCreateInfo) -> Result<Self>
    where
        B: BindableVec + Default + 'static,
    {
        let shader_ci = vk::ShaderModuleCreateInfo::builder()
            .code(&create_info.code[..]);

        Ok(Self {
            shader: unsafe { loader.device.create_shader_module(&shader_ci, None)? },
            entry: create_info.entry,
            bindings: Rc::new(B::default()),
        })
    }

    pub fn descriptor_bindings(&self) -> Vec<vk::DescriptorSetLayoutBinding> {
        self.bindings.bindings()
    }
}

impl Shader for FragmentShader {
    fn shader_flags() -> vk::ShaderStageFlags {
        vk::ShaderStageFlags::FRAGMENT
    }
}

pub trait Vertex {
    fn bindings(&self) -> Vec<vk::VertexInputBindingDescription>;
    fn attributes(&self) -> Vec<vk::VertexInputAttributeDescription>;
}

pub trait BindableVec {
    fn bindings(&self) -> Vec<vk::DescriptorSetLayoutBinding>;
}

pub trait Bindable {
    fn binding(&self) -> vk::DescriptorSetLayoutBinding;
}

impl<T: Bindable> BindableVec for T {
    fn bindings(&self) -> Vec<vk::DescriptorSetLayoutBinding> {
        vec![self.binding()]
    }
}

#[impl_for_tuples(1, 10)]
impl BindableVec for Tuple {
    fn bindings(&self) -> Vec<vk::DescriptorSetLayoutBinding> {
        [for_tuples!( #( Tuple.bindings() ),* )]
            .into_iter()
            .flat_map(|v| v.into_iter())
            .collect()
    }
}

pub trait Shader {
    fn shader_flags() -> vk::ShaderStageFlags;
}