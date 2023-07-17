use anyhow::{anyhow, Result};
use bitflags::bitflags;
use by_address::ByAddress;
use shaderc::ShaderKind;
use std::{collections::HashMap, ops::Deref, rc::Rc, cell::RefCell};

use crate::{
    pipeline::{build_pipeline, build_render_pass},
    prelude::*,
    resources::{ResourceDescription, Layouts, DescriptorSets}, collections::ParitySet,
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

impl Destructible for ShaderModule {
    fn destroy(self, loader: &Loader) {
        self.module.destroy(loader);
    }
}

#[derive(Debug, Clone)]
pub struct ShaderEffect {
    resources: Vec<Rc<ResourceDescription>>,
    shaders: Vec<Identifier>,
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

        let (shaders, _) = modules.clone().into_iter().unzip::<_, _, Vec<_>, Vec<_>>();

        Ok(Self {
            resources,
            shaders,
        })
    }
}

#[derive(Debug, Clone)]
pub struct Pipeline {
    pass: vk::RenderPass,
}

#[derive(Debug, Clone, Default)]
pub struct MaterialSkeleton {
    pub effects: Vec<Identifier>,
}

pub struct MaterialSystemBuilder<'a> {
    loader: &'a Loader,
    resources: HashMap<Identifier, ResourceDescription>,
    shaders: HashMap<Identifier, ShaderModule>,
    effects: HashMap<Identifier, ShaderEffect>,
    skeletons: HashMap<Identifier, MaterialSkeleton>
}

#[derive(Debug, Clone)]
pub struct PipelineData {
    pub local_sets: DescriptorSets,
    pub pipeline: vk::Pipeline,
    pub render_pass: vk::RenderPass,
}

impl Destructible for PipelineData {
    fn destroy(self, loader: &Loader) {
        self.local_sets.destroy(loader);
        self.pipeline.destroy(loader);
        self.render_pass.destroy(loader);
    }
}


#[derive(Debug)]
pub struct MaterialSystem {
    // Copy of material description stuff
    resources: HashMap<Identifier, ResourceDescription>,
    shaders: HashMap<Identifier, ShaderModule>,
    effects: HashMap<Identifier, ShaderEffect>,
    skeletons: HashMap<Identifier, MaterialSkeleton>,

    // Descriptors
    descriptor_pool: RefCell<DescriptorPool>,
    layouts: Layouts,
    global_sets: Option<ParitySet<ManagedDescriptorSet>>,
    pipelines: HashMap<Identifier, PipelineData>
}

impl Destructible for MaterialSystem {
    fn destroy(self, loader: &Loader) {
        self.shaders.into_values().destroy(loader);
        self.descriptor_pool.into_inner().destroy(loader);
        self.global_sets.into_iter().flatten().destroy(loader);
        self.pipelines.into_values().destroy(loader);
    }
}

impl<'a> MaterialSystemBuilder<'a> {
    pub fn new(loader: &'a Loader) -> Self {
        Self {
            loader,
            resources: Default::default(),
            shaders: Default::default(),
            effects: Default::default(),
            skeletons: Default::default(),
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
        id: Identifier,
        identifiers: impl IntoIterator<Item = Identifier> + Clone,
    ) -> Result<Identifier> {
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

        if self.effects.insert(id.clone(), ShaderEffect::new(self.loader, modules)?).is_some() {
            return Err(anyhow!("Effect {} already exists", id))
        }

        Ok(id)
    }

    pub fn register_material(
        &mut self,
        id: Identifier,
        skeleton: MaterialSkeleton
    ) -> Result<Identifier> {
        if self.skeletons.insert(id.clone(), skeleton).is_some() {
            return Err(anyhow!("Skeleton {} already exists", id))
        }

        Ok(id)
    }

    pub fn build(self) -> Result<MaterialSystem> {
        let descriptor_pool = RefCell::new(DescriptorPool::new(self.loader)?);
        let layouts = Layouts::new(self.loader, self.shaders.iter())?;
        let global_sets = match layouts.global_layout {
            Some(layout) => Some(descriptor_pool.borrow_mut().allocate(self.loader, &[layout, layout])?.into_iter().collect()),
            None => None
        };

        Ok(MaterialSystem {
            resources: self.resources,
            shaders: self.shaders,
            effects: self.effects,
            skeletons: self.skeletons,

            descriptor_pool,
            layouts,
            global_sets,
            pipelines: Default::default()
        })
    }
}

impl MaterialSystem {
    pub fn get_effect_pipeline(&mut self, loader: &Loader, id: &Identifier) -> Result<&PipelineData> {
        if !self.pipelines.contains_key(id) {
            let pipeline = self.generate_effect_pipeline(loader, id)?;
            self.pipelines.insert(id.clone(), pipeline);
        }

        Ok(self.pipelines.get(id).unwrap())
    }

    fn generate_effect_pipeline(&self, loader: &Loader, id: &Identifier) -> Result<PipelineData> {
        let effect = self.effects.get(id).ok_or(anyhow!("Effect {} does not exist", id))?;
        let resources = effect.resources.iter().map(|resource| resource.as_ref());
        let shaders = effect.shaders.iter().map(|id| self.shaders.get(id).unwrap());

        let layout = self.layouts.get(id).unwrap();
        let local_sets = DescriptorSets::allocate(loader, &mut self.descriptor_pool.borrow_mut(), layout, None)?;

        let render_pass = build_render_pass(loader, resources.clone())?;
        let pipeline = build_pipeline(loader, render_pass, layout, resources, shaders)?;

        Ok(PipelineData { local_sets, pipeline, render_pass })
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
