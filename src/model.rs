use anyhow::Result;
use memoffset::offset_of;

use crate::pipeline::BindableVertex;
use crate::prelude::*;
use crate::storage::buffer::{self, Buffer};
use crate::storage::descriptors::{Bindable, BindingDescription, DescriptorFrequency, UniformWrite};
use crate::sync::CommandPool;

#[derive(Clone, Copy, Debug)]
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

pub struct ModelBuffer {
    pub vertices: Buffer,
    pub indices: Buffer,
}

impl ModelBuffer {
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

        Ok(Self {
            vertices, indices
        })
    }

    pub fn destroy(self, loader: &Loader) {
        buffer::destroy_buffer(loader, self.vertices);
        buffer::destroy_buffer(loader, self.indices);
    }
}

#[derive(Clone, Copy, Default, Debug)]
pub struct MVP {
    pub model: glam::Mat4,
    pub view: glam::Mat4,
    pub projection: glam::Mat4,
}

impl Bindable for MVP {
    type Write = UniformWrite;

    fn binding(&self) -> BindingDescription {
        BindingDescription {
            descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
            frequency: DescriptorFrequency::Global,
            binding: 0,
            descriptor_count: 1,
            stage_flags: vk::ShaderStageFlags::empty(),
        }
    }

    fn pool_size(&self) -> vk::DescriptorPoolSize {
        vk::DescriptorPoolSize::builder()
            .ty(vk::DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(1)
            .build()
    }
}

pub struct Model {
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u32>,
}
