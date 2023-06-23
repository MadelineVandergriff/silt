use crate::{
    prelude::*,
    storage::{
        descriptors::{build_layout, DescriptorFrequency, Layouts, ShaderBinding, VertexInput}, image::AttachmentType,
    }, loader,
    resources::{
        Buffer,
        Image,
        SampledImage, RedundantSet, UniformBuffer
    }
};
use anyhow::{anyhow, Result};
use bitflags::bitflags;
use by_address::ByAddress;
use itertools::Itertools;
use shaderc::ShaderKind;
use std::{collections::HashMap, rc::Rc, marker::PhantomData, ops::Deref};
use derive_more::{IsVariant};

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
pub struct Pipeline {
    pass: vk::RenderPass,
}

pub struct MaterialSkeleton {
    pub effects: Vec<Rc<ShaderEffect>>,
}

pub struct MaterialSystem<'a> {
    loader: &'a Loader,
    resources: HashMap<Identifier, ResourceDescription>,
    shaders: HashMap<Identifier, ShaderCode>,
    pipelines: HashMap<ByAddress<Rc<ShaderEffect>>, Option<Pipeline>>,
}

impl<'a> MaterialSystem<'a> {
    pub fn new(loader: &'a Loader) -> Self {
        Self {
            loader,
            resources: Default::default(),
            shaders: Default::default(),
            pipelines: Default::default()
        }
    }

    pub fn get_description(&self, id: &Identifier) -> Option<&ResourceDescription> {
        self.resources.get(id)
    }

    pub fn add_basic_uniform<T>(&mut self, id: Identifier, binding: u32, frequency: DescriptorFrequency) -> Result<TypedIdentifier<T>> {
        let desc = ResourceDescription::Uniform {
            binding: ShaderBinding {
                ty: vk::DescriptorType::UNIFORM_BUFFER,
                frequency,
                binding,
                count: 1,
            },
            stride: std::mem::size_of::<T>() as u64,
            elements: 1,
            host_visible: true,
        };

        if self.resources.insert(id.clone(), desc).is_some() {
            return Err(anyhow!("Resource id {} already exists", id))
        };

        Ok(id.into())
    }

    pub fn add_basic_sampled_image(&mut self, id: Identifier, binding: u32, frequency: DescriptorFrequency) -> Result<Identifier> {
        let desc = ResourceDescription::SampledImage {
            binding: ShaderBinding {
                ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                frequency,
                binding,
                count: 1,
            },
        };

        if self.resources.insert(id.clone(), desc).is_some() {
            return Err(anyhow!("Resource id {} already exists", id))
        };

        Ok(id)
    }

    pub fn add_vertex_input<T: VertexInput>(&mut self, id: Identifier) -> Result<Identifier> {
        let desc = T::resource_description();

        if self.resources.insert(id.clone(), desc).is_some() {
            return Err(anyhow!("Resource id {} already exists", id))
        };

        Ok(id)
    }

    pub fn add_shader(&mut self, id: Identifier, code: (Vec<u32>, ShaderKind), resources: impl IntoIterator<Item = Identifier>) -> Result<Identifier> {
        let resources = resources
            .into_iter()
            .map(|id| {
                self.resources.get(&id).map(std::clone::Clone::clone)
            })
            .collect::<Option<Vec<_>>>()
            .ok_or(anyhow!("Identifier doesn't point to valid resource"))?;

        let shader = ShaderCode {
            code: code.0,
            kind: shader_kind_to_shader_stage_flags(code.1),
            resources,
        };

        if self.shaders.insert(id.clone(), shader).is_some() {
            return Err(anyhow!("Resource id {} already exists", id))
        };

        Ok(id)
    }

    pub fn build_uniform<T: Copy>(&self, id: &TypedIdentifier<T>, value: T) -> Result<Named<UniformBuffer<T>>> {
        let desc = match self.get_description(id) {
            None => return Err(anyhow!("Identifier {} does not point to a resource description", id.as_str())),
            Some(desc) if !desc.is_uniform() => return Err(anyhow!("Identifier {} does not point to a uniform resource description", id.as_str())),
            Some(desc) => desc.clone()
        };

        let buffer = UniformBuffer::new(self.loader, &desc.into(), value, Some(id.as_ref().clone()))?;

        Ok(Named {
            inner: buffer,
            id: id.as_ref().clone(),
        })
    }

    pub fn register_effect(
        &mut self,
        modules: impl Into<Vec<ShaderCode>>,
    ) -> Result<Rc<ShaderEffect>> {
        let effect = Rc::new(ShaderEffect::new(self.loader, modules)?);
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
        todo!()
    }
}

pub struct Material {}

#[derive(Debug, Clone, IsVariant)]
pub enum ResourceDescription {
    Uniform {
        binding: ShaderBinding,
        stride: vk::DeviceSize,
        elements: usize,
        host_visible: bool,
    },
    SampledImage {
        binding: ShaderBinding,
    },
    VertexInput {
        bindings: Vec<vk::VertexInputBindingDescription>,
        attributes: Vec<vk::VertexInputAttributeDescription>,
    },
    Attachment {
        ty: AttachmentType,
        format: vk::Format,
    },
}

#[derive(Debug, Clone)]
pub struct TypedResourceDescription<T> {
    inner: ResourceDescription,
    phantom: PhantomData<T>
}

impl<T> Deref for TypedResourceDescription<T> {
    type Target = ResourceDescription;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl<T> From<ResourceDescription> for TypedResourceDescription<T> {
    fn from(inner: ResourceDescription) -> Self {
        Self {
            inner,
            phantom: PhantomData
        }
    }
}

impl ResourceDescription {
    pub fn get_shader_binding(&self) -> Option<&ShaderBinding> {
        match self {
            Self::Uniform { binding, .. } => Some(binding),
            Self::SampledImage { binding, .. } => Some(binding),
            Self::Attachment { ty, .. } => {
                match ty {
                    AttachmentType::Input(binding) => Some(binding),
                    AttachmentType::DepthInput(binding) => Some(binding),
                    _ => None
                }
            },
            _ => None,
        }
    }

    pub fn uniform<T>(binding: u32, frequency: DescriptorFrequency) -> TypedResourceDescription<T> {
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
        }.into()
    }

    pub fn sampled_image(binding: u32, frequency: DescriptorFrequency) -> Self {
        Self::SampledImage {
            binding: ShaderBinding {
                ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                frequency,
                binding,
                count: 1,
            },
        }
    }
}

/*
pub enum Resource<'a> {
    Uniform(RedundantSet<&'a Buffer>),
    SampledImage(&'a SampledImage),
    Attachment(RedundantSet<&'a Image>),
}
*/
