use derive_more::{Constructor, Deref, From, IsVariant, Unwrap};
use std::{marker::PhantomData, rc::Rc};

use crate::{
    prelude::*,
    storage::{
        descriptors::{DescriptorFrequency, ShaderBinding, VertexInput, PartialShaderBinding},
    },
};

#[derive(Debug, Clone, PartialEq)]
pub struct UniformDescription {
    pub id: Identifier,
    pub binding: PartialShaderBinding,
    pub stride: vk::DeviceSize,
    pub elements: usize,
    pub host_visible: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SampledImageDescription {
    pub id: Identifier,
    pub binding: PartialShaderBinding,
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, IsVariant, Unwrap)]
pub enum AttachmentType {
    Color, DepthStencil, Input(PartialShaderBinding), Resolve
}

impl AttachmentType {
    pub fn load_op(&self) -> vk::AttachmentLoadOp {
        match self {
            AttachmentType::Color => vk::AttachmentLoadOp::CLEAR,
            AttachmentType::DepthStencil => vk::AttachmentLoadOp::CLEAR,
            AttachmentType::Input(_) => vk::AttachmentLoadOp::LOAD,
            AttachmentType::Resolve => vk::AttachmentLoadOp::DONT_CARE,
        }
    }

    pub fn store_op(&self) -> vk::AttachmentStoreOp {
        match self {
            AttachmentType::Color => vk::AttachmentStoreOp::STORE,
            AttachmentType::DepthStencil => vk::AttachmentStoreOp::STORE,
            AttachmentType::Input(_) => vk::AttachmentStoreOp::DONT_CARE,
            AttachmentType::Resolve => vk::AttachmentStoreOp::STORE,
        }
    }

    pub fn stencil_load_op(&self, use_stencil: bool) -> vk::AttachmentLoadOp {
        match self {
            AttachmentType::DepthStencil if use_stencil => vk::AttachmentLoadOp::CLEAR,
            _ => vk::AttachmentLoadOp::DONT_CARE
        }
    }

    pub fn stencil_store_op(&self, use_stencil: bool) -> vk::AttachmentStoreOp {
        match self {
            AttachmentType::DepthStencil if use_stencil => vk::AttachmentStoreOp::STORE,
            _ => vk::AttachmentStoreOp::DONT_CARE
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct AttachmentDescription {
    pub id: Identifier,
    pub ty: AttachmentType,
    pub use_stencil: bool,
    pub format: vk::Format,
    pub samples: vk::SampleCountFlags,
    pub final_layout: vk::ImageLayout,
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
        F: FnOnce(&Self) -> Result<R, E>,
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

impl<T> Typed for TypedResourceDescription<T> {
    type Inner = T;
}

impl ResourceDescription {
    pub fn get_shader_binding(&self) -> Option<ShaderBinding> {
        Some(match self {
            Self::Uniform(desc) => {
                ShaderBinding {
                    ty: vk::DescriptorType::UNIFORM_BUFFER,
                    ..desc.binding.as_binding()
                }
            },
            Self::SampledImage(desc) => {
                ShaderBinding {
                    ty: vk::DescriptorType::UNIFORM_BUFFER,
                    ..desc.binding.as_binding()
                }
            },
            Self::Attachment(desc) if desc.ty.is_input() => {
                ShaderBinding {
                    ty: vk::DescriptorType::INPUT_ATTACHMENT,
                    ..desc.ty.unwrap_input().as_binding()
                }
            }
            _ => return None
        })
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
        F: FnOnce(&Rc<Self>) -> Result<R, E>,
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
            binding: PartialShaderBinding {
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
                binding: PartialShaderBinding {
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