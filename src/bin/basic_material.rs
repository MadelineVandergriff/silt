use memoffset::offset_of;
use silt::loader::{LoaderCreateInfo, LoaderHandles};
use silt::material::{ResourceDescription, ShaderOptions, MaterialSystem, MaterialSkeleton};
use silt::resources::UniformBuffer;
use silt::{prelude::*, resources, sampled_image};
use silt::properties::{DeviceFeaturesRequest, DeviceFeatures, ProvidedFeatures};
use silt::{shader, uniform_buffer};
use silt::storage::descriptors::{ShaderBinding, DescriptorFrequency, VertexInput};
use silt::resources::ImageCreateInfo;
use anyhow::Result;
use silt::sync::{QueueRequest, QueueType};

struct Vertex {
    pos: glam::Vec3,
    uv: glam::Vec2,
}

#[allow(dead_code)]
#[derive(Debug, Default, Clone, Copy)]
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

    let vertex_shader = shader!("../../assets/shaders/model_loading.vert", ShaderOptions::HLSL, vertex, *mvp)?;
    let fragment_shader = shader!("../../assets/shaders/model_loading.frag", ShaderOptions::HLSL)?;

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

    let (loader, LoaderHandles { debug_messenger, surface, pdevice, queues }) = Loader::new(loader_ci)?;
    let features = ProvidedFeatures::new(&loader, pdevice);

    //let mvp_buffer = UniformBuffer::new(&loader, &mvp, MVP::default())?;
    uniform_buffer!(mvp_buffer, loader, mvp, MVP::default());

    let image_ci = ImageCreateInfo::default();
    sampled_image!(texture_image, loader, texture, image_ci, features);

    let mut material_system = MaterialSystem::new(&loader);
    let model_loading = material_system.register_effect([vertex_shader, fragment_shader])?;

    let skeleton = MaterialSkeleton {
        effects: vec![model_loading.clone()],
    };

    let pipeline = material_system.build_pipeline(model_loading)?;

    Ok(())
}
