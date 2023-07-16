use anyhow::{anyhow, Result};
use bitflags::bitflags;
use by_address::ByAddress;
use shaderc::ShaderKind;
use std::{collections::HashMap, ops::Deref, rc::Rc};

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

#[derive(Debug)]
pub struct MaterialSystem {
    // Copy of material description stuff
    resources: HashMap<Identifier, ResourceDescription>,
    shaders: HashMap<Identifier, ShaderModule>,
    effects: HashMap<Identifier, ShaderEffect>,
    skeletons: HashMap<Identifier, MaterialSkeleton>,

    // Descriptors
    descriptor_pool: DescriptorPool,
    layouts: Layouts,
    global_sets: Option<ParitySet<ManagedDescriptorSet>>,
    local_sets: HashMap<Identifier, DescriptorSets>
}

impl Destructible for MaterialSystem {
    fn destroy(self, loader: &Loader) {
        self.shaders.into_values().destroy(loader);
        self.descriptor_pool.destroy(loader);
        self.global_sets.into_iter().flatten().destroy(loader);
        self.local_sets.into_values().destroy(loader);
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
        let mut descriptor_pool = DescriptorPool::new(self.loader)?;
        let layouts = Layouts::new(self.loader, self.shaders.iter())?;
        let global_sets = match layouts.global_layout {
            Some(layout) => Some(descriptor_pool.allocate(self.loader, &[layout, layout])?.into_iter().collect()),
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
            local_sets: Default::default()
        })
    }
}

impl MaterialSystem {
    fn generate_effect_pipeline(&mut self, loader: &Loader, id: &Identifier) -> Result<()> {
        let effect = self.effects.get(id).ok_or(anyhow!("Effect {} does not exist", id))?;
        let layout = self.layouts.get(id).unwrap();
        let local_sets = DescriptorSets::allocate(loader, &mut self.descriptor_pool, layout, None);

        Ok(())
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
