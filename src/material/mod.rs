use crate::{
    prelude::*,
    storage::{
        descriptors::{build_layout, DescriptorFrequency, Layouts, ShaderBinding, VertexInput},
        image::{AttachmentType, ImageCreateInfo},
    },
};
use anyhow::{anyhow, Result};
use bitflags::bitflags;
use by_address::ByAddress;
use std::{collections::HashMap, rc::Rc};

bitflags! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct ShaderOptions: u32 {
        const HLSL  = 0b00000001;
        const CACHE = 0b00000010;
    }
}

impl Into<ShaderOptions> for () {
    fn into(self) -> ShaderOptions {
        ShaderOptions::empty()
    }
}

#[derive(Debug, Clone)]
pub struct ShaderCode {
    pub code: Vec<u32>,
    pub kind: vk::ShaderStageFlags,
    pub resources: Vec<ResourceDescription>,
}

#[derive(Debug, Clone)]
pub struct ShaderEffect {
    pub modules: Vec<ShaderCode>,
    pub layouts: Layouts,
}

impl ShaderEffect {
    pub fn new(loader: &Loader, modules: impl Into<Vec<ShaderCode>>) -> Result<Self> {
        let modules: Vec<_> = modules.into();

        Ok(Self {
            layouts: build_layout(loader, &modules)?,
            modules,
        })
    }
}

#[derive(Debug, Clone)]
pub struct Pipeline {}

pub struct MaterialSkeleton {
    pub effects: Vec<Rc<ShaderEffect>>,
}

#[derive(Default)]
pub struct MaterialSystem {
    pipelines: HashMap<ByAddress<Rc<ShaderEffect>>, Option<Pipeline>>,
}

impl MaterialSystem {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register_effect(
        &mut self,
        loader: &Loader,
        modules: impl Into<Vec<ShaderCode>>,
    ) -> Result<Rc<ShaderEffect>> {
        let effect = Rc::new(ShaderEffect::new(loader, modules)?);
        match self.pipelines.insert(ByAddress(effect.clone()), None) {
            Some(_) => Err(anyhow!("Effect already registered")),
            None => Ok(effect),
        }
    }

    pub fn build_material(&mut self, skeleton: MaterialSkeleton) -> Result<Material> {
        let pipelines = skeleton
            .effects
            .into_iter()
            .map(|effect| self.build_pipeline(effect).map(Clone::clone))
            .collect::<Result<Vec<_>>>()?;
        Err(anyhow!(""))
    }

    pub fn build_pipeline(&mut self, effect: Rc<ShaderEffect>) -> Result<&Pipeline> {
        match self
            .pipelines
            .get_mut(&ByAddress(effect.clone()))
            .ok_or(anyhow!("Unknown effect"))?
        {
            Some(pipeline) => Ok(pipeline),
            unbuilt => {
                *unbuilt = Some(Self::build_pipeline_uncached(effect));
                Ok(unbuilt.as_ref().unwrap())
            }
        }
    }

    fn build_pipeline_uncached(effect: Rc<ShaderEffect>) -> Pipeline {
        Pipeline {}
    }
}

pub struct Material {}

#[derive(Debug, Clone)]
pub enum ResourceDescription {
    Uniform {
        binding: ShaderBinding,
        stride: vk::DeviceSize,
        elements: usize,
        host_visible: bool,
    },
    SampledImage {
        binding: ShaderBinding,
        image_info: ImageCreateInfo,
    },
    VertexInput {
        bindings: Vec<vk::VertexInputBindingDescription>,
        attributes: Vec<vk::VertexInputAttributeDescription>,
    },
    Attachment {
        binding: ShaderBinding,
        ty: AttachmentType,
        format: vk::Format,
    },
}

impl ResourceDescription {
    pub fn get_shader_binding(&self) -> Option<&ShaderBinding> {
        match self {
            Self::Uniform { binding, .. } => Some(binding),
            Self::SampledImage { binding, .. } => Some(binding),
            Self::Attachment { binding, .. } => Some(binding),
            _ => None,
        }
    }

    pub fn create_uniform<T>(binding: u32, frequency: DescriptorFrequency) -> Self {
        Self::Uniform {
            binding: ShaderBinding {
                ty: vk::DescriptorType::UNIFORM_BUFFER,
                frequency,
                binding,
                count: 1,
            },
            stride: std::mem::size_of::<T>() as u64,
            elements: 1,
            host_visible: true,
        }
    }
}
