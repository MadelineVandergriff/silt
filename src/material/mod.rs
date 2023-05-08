use crate::{prelude::*, storage::descriptors::{ShaderBinding, Layouts, build_layout}};
use anyhow::Result;

#[derive(Debug, Clone)]
pub enum ShaderOptions {
    HLSL,
    Cache,
    VertexBinding {
        bindings: Vec<vk::VertexInputBindingDescription>,
        attributes: Vec<vk::VertexInputAttributeDescription>
    },
    Empty,
    Mixed(Vec<ShaderOptions>),
}

impl Into<ShaderOptions> for () {
    fn into(self) -> ShaderOptions {
        ShaderOptions::Empty
    }
}

impl ShaderOptions {
    pub fn to_vec(self) -> Vec<ShaderOptions> {
        match self {
            Self::Mixed(vec) => vec,
            Self::Empty => vec![],
            other => vec![other]
        }
    }

    pub fn to_vec_ref(&self) -> Vec<&ShaderOptions> {
        match self {
            Self::Mixed(vec) => vec.iter().collect(),
            Self::Empty => vec![],
            other => vec![other]
        }
    }
}

impl PartialEq for ShaderOptions {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::VertexBinding { bindings: l_bindings, attributes: l_attributes }, Self::VertexBinding { bindings: r_bindings, attributes: r_attributes }) => {
                std::iter::zip(l_bindings, r_bindings)
                    .all(|(l, r)| {
                        l.binding == r.binding &&
                        l.stride == r.stride &&
                        l.input_rate == r.input_rate
                    })
                && std::iter::zip(l_attributes, r_attributes)
                    .all(|(l, r)| {
                        l.binding == r.binding &&
                        l.format == r.format &&
                        l.location == r.location &&
                        l.offset == r.offset
                    })
            },
            (Self::Mixed(l0), Self::Mixed(r0)) => l0 == r0,
            _ => core::mem::discriminant(self) == core::mem::discriminant(other),
        }
    }
}

impl std::ops::BitOr for ShaderOptions {
    type Output = ShaderOptions;

    fn bitor(self, rhs: Self) -> Self::Output {
        Self::Mixed(
            self.to_vec().into_iter().chain(rhs.to_vec()).collect()
        )
    }
}

pub struct ShaderCode {
    pub code: Vec<u32>,
    pub kind: vk::ShaderStageFlags,
    pub layout: Vec<ShaderBinding>
}

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