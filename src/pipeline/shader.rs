pub use crate::macros::ShaderCode;

use crate::storage::descriptors::BindableVec;
use crate::{loader::Loader, prelude::*};
use anyhow::Result;
use std::rc::Rc;

pub struct VertexShader {
    pub shader: vk::ShaderModule,
    bindings: Rc<dyn BindableVec>,
    vertex: Rc<dyn BindableVertex>,
}

impl VertexShader {
    pub fn new<V, B>(loader: &Loader, code: ShaderCode) -> Result<Self>
    where
        V: BindableVertex + Default + 'static,
        B: BindableVec + Default + 'static,
    {
        let shader_ci = vk::ShaderModuleCreateInfo::builder().code(&code[..]);

        Ok(Self {
            shader: unsafe { loader.device.create_shader_module(&shader_ci, None)? },
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
}

impl Shader for VertexShader {
    fn shader_flags(&self) -> vk::ShaderStageFlags {
        vk::ShaderStageFlags::VERTEX
    }

    fn descriptor_bindings(&self) -> Vec<vk::DescriptorSetLayoutBinding> {
        self.bindings
            .bindings()
            .into_iter()
            .enumerate()
            .map(|(i, builder)| {
                builder
                    .binding(i as u32)
                    .stage_flags(vk::ShaderStageFlags::VERTEX)
                    .build()
            })
            .collect()
    }
}

pub struct FragmentShader {
    pub shader: vk::ShaderModule,
    bindings: Rc<dyn BindableVec>,
}

impl FragmentShader {
    pub fn new<B>(loader: &Loader, code: ShaderCode) -> Result<Self>
    where
        B: BindableVec + Default + 'static,
    {
        let shader_ci = vk::ShaderModuleCreateInfo::builder().code(&code[..]);

        Ok(Self {
            shader: unsafe { loader.device.create_shader_module(&shader_ci, None)? },
            bindings: Rc::new(B::default()),
        })
    }
}

impl Shader for FragmentShader {
    fn shader_flags(&self) -> vk::ShaderStageFlags {
        vk::ShaderStageFlags::FRAGMENT
    }
    
    fn descriptor_bindings(&self) -> Vec<vk::DescriptorSetLayoutBinding> {
        self.bindings
            .bindings()
            .into_iter()
            .enumerate()
            .map(|(i, builder)| {
                builder
                    .binding(i as u32)
                    .stage_flags(vk::ShaderStageFlags::FRAGMENT)
                    .build()
            })
            .collect()
    }
}

pub trait BindableVertex {
    fn bindings(&self) -> Vec<vk::VertexInputBindingDescription>;
    fn attributes(&self) -> Vec<vk::VertexInputAttributeDescription>;
}

pub trait Shader {
    fn shader_flags(&self) -> vk::ShaderStageFlags;
    fn descriptor_bindings(&self) -> Vec<vk::DescriptorSetLayoutBinding>;
}

pub struct Shaders {
    pub vertex: VertexShader,
    pub fragment: FragmentShader,
}

impl Shaders {
    pub fn get_slice(&self) -> Vec<&dyn Shader> {
        vec![&self.vertex, &self.fragment]
    }
}