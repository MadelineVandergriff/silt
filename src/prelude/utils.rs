use shaderc::ShaderKind;

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

pub trait Typed {
    type Inner;
}

pub struct ShaderCode {
    pub code: Vec<u32>,
    pub kind: ShaderKind
}