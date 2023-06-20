use std::{sync::Arc, fmt::Display};
use shaderc::ShaderKind;
use derive_more::{Deref, From, Into};
use once_cell::sync::Lazy;

use crate::vk;

pub fn shader_kind_to_shader_stage_flags(kind: ShaderKind) -> vk::ShaderStageFlags {
    match kind {
        ShaderKind::Vertex => vk::ShaderStageFlags::VERTEX,
        ShaderKind::Fragment => vk::ShaderStageFlags::FRAGMENT,
        ShaderKind::Compute => vk::ShaderStageFlags::COMPUTE,
        ShaderKind::Geometry => vk::ShaderStageFlags::GEOMETRY,
        ShaderKind::TessControl => vk::ShaderStageFlags::TESSELLATION_CONTROL,
        ShaderKind::TessEvaluation => vk::ShaderStageFlags::TESSELLATION_EVALUATION,
        ShaderKind::DefaultVertex => vk::ShaderStageFlags::VERTEX,
        ShaderKind::DefaultFragment => vk::ShaderStageFlags::FRAGMENT,
        ShaderKind::DefaultCompute => vk::ShaderStageFlags::COMPUTE,
        ShaderKind::DefaultGeometry => vk::ShaderStageFlags::GEOMETRY,
        ShaderKind::DefaultTessControl => vk::ShaderStageFlags::TESSELLATION_CONTROL,
        ShaderKind::DefaultTessEvaluation => vk::ShaderStageFlags::TESSELLATION_EVALUATION,
        ShaderKind::RayGeneration => vk::ShaderStageFlags::RAYGEN_KHR,
        ShaderKind::AnyHit => vk::ShaderStageFlags::ANY_HIT_KHR,
        ShaderKind::ClosestHit => vk::ShaderStageFlags::CLOSEST_HIT_KHR,
        ShaderKind::Miss => vk::ShaderStageFlags::MISS_KHR,
        ShaderKind::Intersection => vk::ShaderStageFlags::INTERSECTION_KHR,
        ShaderKind::Callable => vk::ShaderStageFlags::CALLABLE_KHR,
        ShaderKind::DefaultRayGeneration => vk::ShaderStageFlags::RAYGEN_KHR,
        ShaderKind::DefaultAnyHit => vk::ShaderStageFlags::ANY_HIT_KHR,
        ShaderKind::DefaultClosestHit => vk::ShaderStageFlags::CLOSEST_HIT_KHR,
        ShaderKind::DefaultMiss => vk::ShaderStageFlags::MISS_KHR,
        ShaderKind::DefaultIntersection => vk::ShaderStageFlags::INTERSECTION_KHR,
        ShaderKind::DefaultCallable => vk::ShaderStageFlags::CALLABLE_KHR,
        ShaderKind::Task => vk::ShaderStageFlags::TASK_EXT,
        ShaderKind::Mesh => vk::ShaderStageFlags::MESH_EXT,
        ShaderKind::DefaultTask => vk::ShaderStageFlags::TASK_EXT,
        ShaderKind::DefaultMesh => vk::ShaderStageFlags::MESH_EXT,
        _ => panic!("Unknown shader kind")
    }
}

#[derive(Debug, Clone, Hash, PartialEq, Eq, Deref, From, Into)]
#[deref(forward)]
pub struct Identifier(Arc<str>);

pub static NULL_ID: Lazy<Identifier> = Lazy::new(|| Identifier::new("undefined id"));

impl AsRef<str> for Identifier {
    fn as_ref(&self) -> &str {
        &self.0
    }
}

impl Display for Identifier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl Identifier {
    pub fn new(id: impl AsRef<str>) -> Self {
        Self(Arc::from(id.as_ref()))
    }

    pub fn as_str(&self) -> &str {
        &self
    }
}

#[macro_export]
macro_rules! id {
    ($id: expr) => {
        crate::prelude::Identifier::new($id)
    };
}