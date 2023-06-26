use anyhow::{anyhow, Result};
use bitflags::bitflags;
use by_address::ByAddress;
use shaderc::ShaderKind;
use std::{collections::HashMap, rc::Rc};

use crate::{
    pipeline::build_render_pass,
    prelude::*,
    resources::ResourceDescription,
    storage::descriptors::{build_layout, Layouts},
};

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
    pub resources: Vec<Rc<ResourceDescription>>,
    pub layouts: Layouts,
}

impl ShaderEffect {
    pub fn new<'a, I>(loader: &Loader, modules: I) -> Result<Self>
    where
        I: IntoIterator<Item = &'a ShaderCode> + Clone + 'a,
    {
        let resources = modules
            .clone()
            .into_iter()
            .flat_map(|s| s.resources.iter().cloned())
            .collect();

        Ok(Self {
            layouts: build_layout(loader, modules)?,
            resources,
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
            .map(|id| {
                self.shaders
                    .get(&id)
                    .ok_or_else(|| anyhow!("Identifier {} does not point to a valid shader", id))
            })
            .collect::<Result<Vec<_>>>()?;

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
                *unbuilt = Some(Self::build_pipeline_uncached(self.loader, effect)?);
                Ok(unbuilt.as_ref().unwrap())
            }
        }
    }

    fn build_pipeline_uncached(loader: &Loader, effect: Rc<ShaderEffect>) -> Result<Pipeline> {
        let render_pass = build_render_pass(loader, effect.resources.iter().cloned())?;

        Err(anyhow!("todo"))
    }
}

pub struct Material {}

/*
pub enum Resource<'a> {
    Uniform(RedundantSet<&'a Buffer>),
    SampledImage(&'a SampledImage),
    Attachment(RedundantSet<&'a Image>),
}
*/
