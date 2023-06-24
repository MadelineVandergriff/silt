use crate::{
    loader,
    prelude::*,
    resources::{Buffer, Image, RedundantSet, SampledImage, UniformBuffer},
    storage::{
        descriptors::{build_layout, DescriptorFrequency, Layouts, ShaderBinding, VertexInput},
        image::AttachmentType,
    }, id,
};
use anyhow::{anyhow, Result};
use bitflags::bitflags;
use by_address::ByAddress;
use derive_more::{Constructor, Deref, From, IsVariant, Unwrap};
use itertools::Itertools;
use shaderc::ShaderKind;
use std::{collections::HashMap, marker::PhantomData, ops::Deref, rc::Rc};

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
    pub resources: Vec<Rc<ResourceDescription>>,
}

#[derive(Debug, Clone)]
pub struct ShaderEffect {
    pub identifiers: Vec<Identifier>,
    pub layouts: Layouts,
}

impl ShaderEffect {
    pub fn new<'a>(loader: &Loader, modules: impl IntoIterator<Item = (Identifier, &'a ShaderCode)>) -> Result<Self> {
        let (identifiers, modules): (Vec<_>, Vec<_>) = modules.into_iter().unzip();

        Ok(Self {
            layouts: build_layout(loader, modules)?,
            identifiers,
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
            pipelines: Default::default(),
        }
    }

    pub fn get_description(&self, id: &Identifier) -> Option<&ResourceDescription> {
        self.resources.get(id)
    }

    pub fn add_shader(
        &mut self,
        id: Identifier,
        code: (Vec<u32>, ShaderKind),
        resources: impl IntoIterator<Item = Rc<ResourceDescription>>,
    ) -> Result<Identifier> {
        let shader = ShaderCode {
            code: code.0,
            kind: shader_kind_to_shader_stage_flags(code.1),
            resources: resources.into_iter().collect(),
        };

        if self.shaders.insert(id.clone(), shader).is_some() {
            return Err(anyhow!("Resource id {} already exists", id));
        };

        Ok(id)
    }

    pub fn register_effect(
        &mut self,
        identifiers: impl IntoIterator<Item = Identifier> + Clone,
    ) -> Result<Rc<ShaderEffect>> {
        let modules = identifiers
            .clone()
            .into_iter()
            .map(|id| self.shaders.get(&id).ok_or_else(|| anyhow!("Identifier {} does not point to a valid shader", id)))
            .collect::<Result<Vec<_>>>()?;

        let effect = Rc::new(ShaderEffect::new(self.loader, std::iter::zip(identifiers, modules))?);
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

#[derive(Debug, Clone, PartialEq)]
pub struct UniformDescription {
    pub id: Identifier,
    pub binding: ShaderBinding,
    pub stride: vk::DeviceSize,
    pub elements: usize,
    pub host_visible: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SampledImageDescription {
    pub id: Identifier,
    pub binding: ShaderBinding,
}

#[derive(Debug, Clone)]
pub struct VertexInputDescription {
    pub id: Identifier,
    pub bindings: Vec<vk::VertexInputBindingDescription>,
    pub attributes: Vec<vk::VertexInputAttributeDescription>,
}

// Why doesn't ash implement PartialEq for these structs????
impl PartialEq for VertexInputDescription {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
            && self.bindings.len() == other.bindings.len()
            && self.attributes.len() == other.attributes.len()
            && std::iter::zip(&self.bindings, &other.bindings).all(|(a, b)| {
                a.binding == b.binding && a.stride == b.stride && a.input_rate == b.input_rate
            })
            && std::iter::zip(&self.attributes, &other.attributes).all(|(a, b)| {
                a.binding == b.binding
                    && a.format == b.format
                    && a.location == b.location
                    && a.offset == b.offset
            })
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct AttachmentDescription {
    pub id: Identifier,
    pub ty: AttachmentType,
    pub format: vk::Format,
}

#[derive(Debug, Clone, PartialEq, IsVariant, Unwrap, From)]
pub enum ResourceDescription {
    Uniform(UniformDescription),
    SampledImage(SampledImageDescription),
    VertexInput(VertexInputDescription),
    Attachment(AttachmentDescription),
}

impl Identified for ResourceDescription {
    fn id(&self) -> &Identifier {
        match self {
            Self::Uniform(desc) => &desc.id,
            Self::SampledImage(desc) => &desc.id,
            Self::VertexInput(desc) => &desc.id,
            Self::Attachment(desc) => &desc.id,
        }
    }
}

#[derive(Debug, Clone, Deref)]
pub struct TypedResourceDescription<T> {
    #[deref(forward)]
    inner: Rc<ResourceDescription>,
    phantom: PhantomData<T>,
}

impl<T> TypedResourceDescription<T> {
    pub fn bind<F, R>(&self, f: F) -> Resource<R>
    where
        F: FnOnce(&Self) -> R,
    {
        Resource {
            resource: f(self),
            description: self.inner.clone(),
        }
    }

    pub fn bind_result<F, R, E>(&self, f: F) -> Result<Resource<R>, E>
    where
        F: FnOnce(&Self) -> Result<R, E>
    {
        Ok(Resource {
            resource: f(self)?,
            description: self.inner.clone(),
        })
    }
}

impl<T> From<Rc<ResourceDescription>> for TypedResourceDescription<T> {
    fn from(inner: Rc<ResourceDescription>) -> Self {
        Self {
            inner,
            phantom: PhantomData,
        }
    }
}

impl<T> From<TypedResourceDescription<T>> for Rc<ResourceDescription> {
    fn from(description: TypedResourceDescription<T>) -> Self {
        description.inner
    }
}

pub trait Typed {
    type Inner;
}

impl<T> Typed for TypedResourceDescription<T> {
    type Inner = T;
}

impl ResourceDescription {
    pub fn get_shader_binding(&self) -> Option<&ShaderBinding> {
        match self {
            Self::Uniform(desc) => Some(&desc.binding),
            Self::SampledImage(desc) => Some(&desc.binding),
            Self::Attachment(desc) => match &desc.ty {
                AttachmentType::Input(binding) => Some(binding),
                AttachmentType::DepthInput(binding) => Some(binding),
                _ => None,
            },
            _ => None,
        }
    }

    pub fn bind<F, R>(self: &Rc<Self>, f: F) -> Resource<R>
    where
        F: FnOnce(&Rc<Self>) -> R,
    {
        Resource {
            resource: f(self),
            description: self.clone(),
        }
    }

    pub fn bind_result<F, R, E>(self: &Rc<Self>, f: F) -> Result<Resource<R>, E>
    where
        F: FnOnce(&Rc<Self>) -> Result<R, E>
    {
        Ok(Resource {
            resource: f(self)?,
            description: self.clone(),
        })
    }

    pub fn uniform<T>(
        id: Identifier,
        binding: u32,
        frequency: DescriptorFrequency,
    ) -> TypedResourceDescription<T> {
        Rc::new(Self::from(UniformDescription {
            id,
            binding: ShaderBinding {
                ty: vk::DescriptorType::UNIFORM_BUFFER,
                frequency,
                binding,
                count: 1,
            },
            stride: std::mem::size_of::<T>() as u64,
            elements: 1,
            host_visible: true,
        }))
        .into()
    }

    pub fn sampled_image(id: Identifier, binding: u32, frequency: DescriptorFrequency) -> Rc<Self> {
        Rc::new(
            SampledImageDescription {
                id,
                binding: ShaderBinding {
                    ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                    frequency,
                    binding,
                    count: 1,
                },
            }
            .into(),
        )
    }

    pub fn vertex_input<T: VertexInput>(id: Identifier) -> Rc<Self> {
        Rc::new(
            VertexInputDescription {
                id,
                bindings: T::bindings(),
                attributes: T::attributes(),
            }
            .into(),
        )
    }
}

#[macro_export]
macro_rules! resources {
    ($($desc: expr),*) => {
        [$($desc.clone().into()),*]
    };
}

#[derive(Debug, Clone, Constructor)]
pub struct Resource<T> {
    pub resource: T,
    pub description: Rc<ResourceDescription>,
}

/*
pub enum Resource<'a> {
    Uniform(RedundantSet<&'a Buffer>),
    SampledImage(&'a SampledImage),
    Attachment(RedundantSet<&'a Image>),
}
*/
