use crate::{prelude::*, storage::descriptors::{ShaderBinding, Layouts, build_layout}};
use anyhow::{anyhow, Result};
use std::{rc::{Weak, Rc}, collections::HashMap};
use by_address::ByAddress;

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

#[derive(Debug, Clone)]
pub struct ShaderCode {
    pub code: Vec<u32>,
    pub kind: vk::ShaderStageFlags,
    pub layout: Vec<ShaderBinding>
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

}

pub struct MaterialSkeleton {
    pub effects: Vec<Rc<ShaderEffect>>
}

#[derive(Default)]
pub struct MaterialSystem {
    pipelines: HashMap<ByAddress<Rc<ShaderEffect>>, Option<Pipeline>>,
}

impl MaterialSystem {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register_effect(&mut self, loader: &Loader, modules: impl Into<Vec<ShaderCode>>) -> Result<Rc<ShaderEffect>> {
        let effect = Rc::new(ShaderEffect::new(loader, modules)?);
        match self.pipelines.insert(ByAddress(effect.clone()), None) {
            Some(_) => Err(anyhow!("Effect already registered")),
            None => Ok(effect)
        }
    }

    pub fn build_material(&mut self, skeleton: MaterialSkeleton) -> Result<Material> {
        let pipelines = skeleton
            .effects
            .into_iter()
            .map(|effect| {
                self.build_pipeline(effect).map(Clone::clone)
            })
            .collect::<Result<Vec<_>>>()?;
        Err(anyhow!(""))
    }

    pub fn build_pipeline(&mut self, effect: Rc<ShaderEffect>) -> Result<&Pipeline> {
        match self.pipelines.get_mut(&ByAddress(effect.clone())).ok_or(anyhow!("Unknown effect"))? {
            Some(pipeline) => Ok(pipeline),
            unbuilt => {
                *unbuilt = Some(Self::build_pipeline_uncached(effect));
                Ok(unbuilt.as_ref().unwrap())
            },
        }
    }

    fn build_pipeline_uncached(effect: Rc<ShaderEffect>) -> Pipeline {
        Pipeline {
            
        }
    }
}

pub struct Material {

}