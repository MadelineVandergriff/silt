use anyhow::Result;
use memoffset::offset_of;
use silt::loader::{LoaderCreateInfo, LoaderHandles};
use silt::material::{MaterialSkeleton, MaterialSystem, ShaderOptions};
use silt::prelude::*;
use silt::properties::{DeviceFeatures, DeviceFeaturesRequest, ProvidedFeatures};
use silt::resources::{UniformBuffer, ResourceDescription, VertexInput};
use silt::resources::{ImageCreateInfo, ImageFile, SampledImage};
use silt::sync::{CommandPool, QueueRequest, QueueType};
use silt::{compile, id, resources};

struct Vertex {
    pos: glam::Vec3,
    color: glam::Vec3,
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
                .location(1)
                .format(vk::Format::R32G32B32_SFLOAT)
                .offset(offset_of!(Vertex, color) as u32)
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
    let loader_ci = LoaderCreateInfo {
        width: 1920,
        height: 1080,
        title: "Basic Material".into(),
        device_features: DeviceFeaturesRequest {
            required: DeviceFeatures::SAMPLER_ANISOTROPY,
            prefered: DeviceFeatures::empty(),
        },
        queue_requests: vec![QueueRequest {
            ty: QueueType::Graphics,
            count: 1,
        }],
    };

    let (
        loader,
        LoaderHandles {
            debug_messenger,
            surface,
            pdevice,
            queues,
        },
    ) = Loader::new(loader_ci)?;
    let features = ProvidedFeatures::new(&loader, pdevice);
    let pool = CommandPool::new(
        &loader,
        &queues[0],
        vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER,
    )?;

    let mut materials = MaterialSystem::new(&loader);

    let vertex = ResourceDescription::vertex_input::<Vertex>(id!("Pos/UV Vertex"));
    let mvp =
        ResourceDescription::uniform::<MVP>(id!("MVP Uniform"), 0, vk::DescriptorFrequency::Global);
    let texture =
        ResourceDescription::sampled_image(id!("Texture Image"), 1, vk::DescriptorFrequency::Global);

    let vertex_shader = materials.add_shader(
        id!("MVP Vertex Pass"),
        compile!(
            "../../assets/shaders/model_loading.vert",
            ShaderOptions::HLSL
        )?,
        resources!(vertex, mvp),
    )?;

    let fragment_shader = materials.add_shader(
        id!("Unlit Texture Pass"),
        compile!(
            "../../assets/shaders/model_loading.frag",
            ShaderOptions::HLSL
        )?,
        resources!(texture),
    )?;

    let mvp_buffer = mvp.bind_result(|description| {
        UniformBuffer::new(
            &loader,
            description,
            Default::default(),
            Some(description.id().clone()),
        )
    })?;

    let texture_image = texture.bind_result(|_| {
        ImageFile::new("assets/textures/viking_room.png")?.upload_to_gpu(&loader, features, &pool)
    })?;

    let model_loading = materials.register_effect([vertex_shader, fragment_shader])?;

    let skeleton = MaterialSkeleton {
        effects: vec![model_loading.clone()],
    };

    let pipeline = materials.build_pipeline(model_loading)?;

    Ok(())
}
