use memoffset::offset_of;
use silt::loader::LoaderCreateInfo;
use silt::material::{ResourceDescription, ShaderOptions, MaterialSystem, MaterialSkeleton};
use silt::{prelude::*, resources};
use silt::properties::{DeviceFeaturesRequest, DeviceFeatures};
use silt::shader;
use silt::storage::descriptors::{ShaderBinding, DescriptorFrequency, VertexInput};
use silt::storage::image::ImageCreateInfo;
use anyhow::Result;
use silt::sync::{QueueRequest, QueueType};

struct Vertex {
    pos: glam::Vec3,
    uv: glam::Vec2,
}

#[allow(dead_code)]
struct MVP {
    model: glam::Mat4,
    view: glam::Mat4,
    proj: glam::Mat4,
}

impl VertexInput for Vertex {
    fn bindings() -> Vec<vk::VertexInputBindingDescription> {
        vec![vk::VertexInputBindingDescription::builder()
            .binding(0)
            .stride(std::mem::size_of::<Self>() as u32)
            .input_rate(vk::VertexInputRate::VERTEX)
            .build()]
    }

    fn attributes() -> Vec<vk::VertexInputAttributeDescription> {
        vec![
            vk::VertexInputAttributeDescription::builder()
                .binding(0)
                .location(0)
                .format(vk::Format::R32G32B32_SFLOAT)
                .offset(offset_of!(Vertex, pos) as u32)
                .build(),
            vk::VertexInputAttributeDescription::builder()
                .binding(0)
                .location(2)
                .format(vk::Format::R32G32_SFLOAT)
                .offset(offset_of!(Vertex, uv) as u32)
                .build(),
        ]
    }
}

fn main() -> Result<()> {
    let vertex = Vertex::resource_description();
    let mvp = ResourceDescription::uniform::<MVP>(0, DescriptorFrequency::Global);
    let texture = ResourceDescription::sampled_image(1, DescriptorFrequency::Global);

    let vertex_shader = shader!("../../assets/shaders/model_loading.vert", vec![vertex, mvp], ShaderOptions::HLSL)?;
    let fragment_shader = shader!("../../assets/shaders/model_loading.frag", vec![texture], ShaderOptions::HLSL)?;

    let loader_ci = LoaderCreateInfo {
        width: 1920,
        height: 1080,
        title: "Basic Material".into(),
        device_features: DeviceFeaturesRequest {
            required: DeviceFeatures::SAMPLER_ANISOTROPY,
            prefered: DeviceFeatures::empty(),
        },
        queue_requests: vec![
            QueueRequest { ty: QueueType::Graphics, count: 1 }
        ],
    };

    let (loader, _) = Loader::new(loader_ci)?;

    let mvp_buffer = resources::Buffer::new

    let mut material_system = MaterialSystem::new(&loader);
    let model_loading = material_system.register_effect([vertex_shader, fragment_shader])?;

    let skeleton = MaterialSkeleton {
        effects: vec![model_loading.clone()],
    };

    let pipeline = material_system.build_pipeline(model_loading)?;

    Ok(())
}
