use memoffset::offset_of;
use silt::material::{ResourceDescription, ShaderOptions};
use silt::prelude::*;
use silt::shader;
use silt::storage::descriptors::{ShaderBinding, DescriptorFrequency};
use silt::storage::image::ImageCreateInfo;

struct Vertex {
    pos: glam::Vec3,
    uv: glam::Vec2,
}

impl Vertex {
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

fn main() {
    let vertex = ResourceDescription::VertexInput {
        bindings: Vertex::bindings(),
        attributes: Vertex::attributes(),
    };

    let texture = ResourceDescription::SampledImage {
        binding: ShaderBinding {
            ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
            frequency: DescriptorFrequency::Global,
            binding: 0,
            count: 1,
        },
        image_info: ImageCreateInfo {
            width: todo!(),
            height: todo!(),
            mip_levels: todo!(),
            format: todo!(),
            tiling: todo!(),
            usage: todo!(),
            location: todo!(),
            samples: todo!(),
            view_aspect: todo!(),
            name: todo!(),
        },
    };
}
