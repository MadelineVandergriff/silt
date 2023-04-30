use anyhow::Result;
use memoffset::offset_of;

use crate::pipeline::BindableVertex;
use crate::prelude::*;
use crate::storage::buffer::{self, Buffer, BoundBuffer, get_bound_buffer};
use crate::storage::descriptors::{Bindable, BindingDescription, DescriptorFrequency};
use crate::sync::CommandPool;

#[derive(Clone, Copy, Debug, Default)]
pub struct Vertex {
    pub pos: glam::Vec3,
    pub color: glam::Vec3,
    pub tex_coord: glam::Vec2,
}

impl BindableVertex for Vertex {
    fn bindings(&self) -> Vec<vk::VertexInputBindingDescription> {
        vec![vk::VertexInputBindingDescription::builder()
            .binding(0)
            .stride(std::mem::size_of::<Self>() as u32)
            .input_rate(vk::VertexInputRate::VERTEX)
            .build()]
    }

    fn attributes(&self) -> Vec<vk::VertexInputAttributeDescription> {
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
                .offset(offset_of!(Vertex, tex_coord) as u32)
                .build(),
        ]
    }
}

#[derive(Clone, Copy, Default, Debug)]
pub struct MVP {
    pub model: glam::Mat4,
    pub view: glam::Mat4,
    pub projection: glam::Mat4,
}

impl Bindable for MVP {
    fn binding(&self) -> BindingDescription {
        BindingDescription {
            ty: vk::DescriptorType::UNIFORM_BUFFER,
            frequency: DescriptorFrequency::Global,
            binding: 0,
            descriptor_count: 1,
            stage_flags: vk::ShaderStageFlags::empty(),
        }
    }
}

pub struct Model {
    pub vertices: Buffer,
    pub indices: Buffer,
    pub mvp: BoundBuffer<MVP>
}

impl Destructible for Model {
    fn destroy(self, loader: &Loader) {
        self.vertices.destroy(loader);
        self.indices.destroy(loader);
        self.mvp.destroy(loader);
    }
}

impl Model {
    pub fn create(loader: &Loader, pool: &CommandPool, vertices: &[Vertex], indices: &[u32]) -> Result<Self> {
        let vertices = buffer::upload_to_gpu(
            loader,
            pool,
            vertices,
            vk::BufferUsageFlags::VERTEX_BUFFER,
            Some("Vertex Buffer"),
        )?;

        let indices = buffer::upload_to_gpu(
            loader,
            pool,
            indices,
            vk::BufferUsageFlags::VERTEX_BUFFER,
            Some("Index Buffer"),
        )?;

        let mvp = get_bound_buffer(loader, vk::BufferUsageFlags::UNIFORM_BUFFER)?;

        Ok(Self {
            vertices, indices, mvp
        })
    }
}

