use anyhow::{anyhow, Result};
use bitflags::bitflags;
use by_address::ByAddress;
use shaderc::ShaderKind;
use std::{collections::HashMap, ops::Deref, rc::Rc};

use crate::{
    pipeline::{build_pipeline, build_render_pass},
    prelude::*,
    resources::{ResourceDescription, Layouts, build_layout},
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
pub struct ShaderModule {
    pub module: vk::ShaderModule,
    pub stage_flags: vk::ShaderStageFlags,
    pub resources: Vec<Rc<ResourceDescription>>,
}

#[derive(Debug, Clone)]
pub struct ShaderEffect {
    pub resources: Vec<Rc<ResourceDescription>>,
    pub layouts: Layouts,
    pub shaders: Vec<Identifier>,
}

impl ShaderEffect {
    pub fn new<'a, I>(loader: &Loader, modules: I) -> Result<Self>
    where
        I: IntoIterator<Item = (Identifier, &'a ShaderModule)> + Clone + 'a,
    {
        let resources = modules
            .clone()
            .into_iter()
            .flat_map(|(_, s)| s.resources.iter().cloned())
            .collect();

        let (shaders, modules) = modules.into_iter().unzip::<_, _, Vec<_>, Vec<_>>();

        Ok(Self {
            layouts: todo!() /*build_layout(loader, modules)?*/,
            resources,
            shaders,
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
    shaders: HashMap<Identifier, ShaderModule>,
    pipelines: HashMap<ByAddress<Rc<ShaderEffect>>, Pipeline>,
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
        code: ShaderCode,
        resources: impl IntoIterator<Item = Rc<ResourceDescription>>,
    ) -> Result<Identifier> {
        let stage_flags = shader_kind_to_shader_stage_flags(code.kind);

        let create_info = vk::ShaderModuleCreateInfo::builder().code(&code.code);

        let module = unsafe {
            self.loader
                .device
                .create_shader_module(&create_info, None)?
        };

        let shader = ShaderModule {
            module,
            stage_flags,
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
                    .get_key_value(&id)
                    .map(|(id, shader)| (id.clone(), shader))
                    .ok_or_else(|| anyhow!("Identifier {} does not point to a valid shader", id))
            })
            .collect::<Result<Vec<_>>>()?;

        Ok(Rc::new(ShaderEffect::new(self.loader, modules)?))
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
        let key = ByAddress(effect.clone());
        let pipeline = self.build_pipeline_uncached(effect)?;
        self.pipelines.insert(key.clone(), pipeline);
        Ok(self.pipelines.get(&key).unwrap())
    }

    fn build_pipeline_uncached(&self, effect: Rc<ShaderEffect>) -> Result<Pipeline> {
        let resources = effect.resources.iter().map(|resource| resource.deref());
        let shaders = effect
            .shaders
            .iter()
            .map(|id| self.shaders.get(id).unwrap());

        let render_pass = build_render_pass(&self.loader, resources.clone())?;
        let pipeline = build_pipeline(
            &self.loader,
            render_pass,
            &effect.layouts,
            resources.clone(),
            shaders,
        )?;

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
