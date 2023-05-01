use std::path::PathBuf;

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
    pub vertex_buffer: Buffer,
    pub index_buffer: Buffer,
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u32>,
    pub mvp: BoundBuffer<MVP>
}

impl Destructible for Model {
    fn destroy(self, loader: &Loader) {
        self.vertex_buffer.destroy(loader);
        self.index_buffer.destroy(loader);
        self.mvp.destroy(loader);
    }
}

impl Model {
    pub fn create(loader: &Loader, pool: &CommandPool, vertices: Vec<Vertex>, indices: Vec<u32>) -> Result<Self> {
        let vertex_buffer = buffer::upload_to_gpu(
            loader,
            pool,
            &vertices,
            vk::BufferUsageFlags::VERTEX_BUFFER,
            Some("Vertex Buffer"),
        )?;

        let index_buffer = buffer::upload_to_gpu(
            loader,
            pool,
            &indices,
            vk::BufferUsageFlags::VERTEX_BUFFER,
            Some("Index Buffer"),
        )?;

        let mvp = get_bound_buffer(loader, vk::BufferUsageFlags::UNIFORM_BUFFER)?;

        Ok(Self {
            vertex_buffer, index_buffer, vertices, indices, mvp
        })
    }

    pub fn load(loader: &Loader, pool: &CommandPool, path: PathBuf) -> Result<Self> {
        let (models, _) =
            tobj::load_obj(path, &tobj::GPU_LOAD_OPTIONS)?;
        let mesh = &models[0].mesh;
    
        let vertices = std::iter::zip(
            mesh.positions.chunks_exact(3),
            mesh.texcoords.chunks_exact(2),
        )
        .map(|(pos, uv)| Vertex {
            pos: glam::vec3(pos[0], pos[1], pos[2]),
            color: glam::Vec3::ZERO,
            tex_coord: glam::vec2(uv[0], 1. - uv[1]),
        })
        .collect::<Vec<_>>();
    
        println!("Vertex count: [{}]", vertices.len());
        println!("Index count: [{}]", mesh.indices.len());
    
        Self::create(loader, pool, vertices, mesh.indices.clone())
    }
}

