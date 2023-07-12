use anyhow::{anyhow, Result};
use std::{
    cell::{Cell, RefCell},
    ops::Deref,
};

use super::{ResourceDescription, TypedResourceDescription, UniformDescription};
use crate::collections::{ParitySet, Parity};
use crate::{id, prelude::*, resources::Image, sync::CommandPool};

#[derive(Clone, Debug)]
pub struct Buffer {
    pub buffer: vk::Buffer,
    pub allocation: vk::Allocation,
    pub size: vk::DeviceSize,
}

impl Destructible for Buffer {
    fn destroy(self, loader: &Loader) {
        self.buffer.destroy(loader);
        self.allocation.destroy(loader);
    }
}

#[derive(Debug, Clone)]
pub struct BufferCreateInfo {
    pub size: vk::DeviceSize,
    pub name: Identifier,
    pub usage: vk::BufferUsageFlags,
    pub location: vk::MemoryLocation,
}

impl Buffer {
    pub fn new(loader: &Loader, create_info: BufferCreateInfo) -> Result<Self> {
        let buffer_create_info = vk::BufferCreateInfo::builder()
            .size(create_info.size)
            .usage(create_info.usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let buffer = unsafe { loader.device.create_buffer(&buffer_create_info, None)? };
        let requirements = unsafe { loader.device.get_buffer_memory_requirements(buffer) };

        let allocation_create_info = vk::AllocationCreateInfo {
            name: create_info.name.as_str(),
            requirements,
            location: create_info.location,
            linear: true,
            allocation_scheme: vk::AllocationScheme::GpuAllocatorManaged,
        };

        let allocation = loader.allocator.allocate(&allocation_create_info)?;
        unsafe {
            loader.device.bind_buffer_memory(
                buffer,
                loader.allocator.get_memory(allocation)?,
                loader.allocator.get_offset(allocation)?,
            )?
        };

        Ok(Self {
            buffer,
            allocation,
            size: requirements.size,
        })
    }

    pub fn copy_to_buffer(
        &self,
        loader: &Loader,
        pool: &CommandPool,
        dst: &Buffer,
        region: vk::BufferCopy,
    ) -> Result<()> {
        if region.size + region.src_offset > self.size {
            return Err(anyhow!(
                "region exceeded src bounds: [{:?}] [{:?}]",
                region,
                self
            ));
        }

        if region.size + region.dst_offset > dst.size {
            return Err(anyhow!(
                "region exceeded dst bounds: [{:?}] [{:?}]",
                region,
                dst
            ));
        }

        pool.execute_one_time_commands(loader, |loader, cmd| unsafe {
            loader.device.cmd_copy_buffer(
                cmd,
                self.buffer,
                dst.buffer,
                std::slice::from_ref(&region),
            );
        })
    }

    pub fn copy_to_entire_buffer(
        &self,
        loader: &Loader,
        pool: &CommandPool,
        dst: &Buffer,
    ) -> Result<()> {
        if self.size != dst.size {
            return Err(anyhow!("src and dst mismatch: [{:?}] [{:?}]", self, dst));
        }

        let region = vk::BufferCopy::builder().size(self.size).build();
        self.copy_to_buffer(loader, pool, dst, region)
    }

    pub fn copy_to_image(
        &self,
        loader: &Loader,
        pool: &CommandPool,
        dst: &Image,
        region: vk::BufferImageCopy,
    ) -> Result<()> {
        let region_volume = vk::Volume3D::from(region.image_extent).offset_by(region.image_offset);
        if !vk::Volume3D::from(dst.size).contains(&region_volume) {
            return Err(anyhow!(
                "Region extents past image bounds: [{:?}] [{:?}]",
                region,
                dst
            ));
        }

        pool.execute_one_time_commands(loader, |loader, cmd| unsafe {
            loader.device.cmd_copy_buffer_to_image(
                cmd,
                self.buffer,
                dst.image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                std::slice::from_ref(&region),
            );
        })
    }

    pub fn copy_to_entire_image(
        &self,
        loader: &Loader,
        pool: &CommandPool,
        dst: &Image,
    ) -> Result<()> {
        let region = vk::BufferImageCopy::builder()
            .buffer_offset(0)
            .buffer_row_length(0)
            .buffer_image_height(0)
            .image_subresource(
                vk::ImageSubresourceLayers::builder()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .mip_level(0)
                    .base_array_layer(0)
                    .layer_count(1)
                    .build(),
            )
            .image_offset(vk::Offset3D::default())
            .image_extent(dst.size);

        self.copy_to_image(loader, pool, dst, *region)
    }

    pub fn upload_to_gpu<T: Copy>(
        loader: &Loader,
        pool: &CommandPool,
        data: &[T],
        usage: vk::BufferUsageFlags,
        name: Identifier,
    ) -> Result<Self> {
        let staging_ci = BufferCreateInfo {
            size: std::mem::size_of_val(data) as u64,
            name: NULL_ID.clone(),
            usage: usage | vk::BufferUsageFlags::TRANSFER_SRC,
            location: vk::MemoryLocation::CpuToGpu,
        };

        let staging = Self::new(loader, staging_ci)?;
        unsafe { get_align(loader, &staging).copy_from_slice(data) };

        let buffer_ci = BufferCreateInfo {
            size: std::mem::size_of_val(data) as u64,
            name,
            usage: usage | vk::BufferUsageFlags::TRANSFER_DST,
            location: vk::MemoryLocation::GpuOnly,
        };

        let buffer = Self::new(loader, buffer_ci)?;

        staging.copy_to_entire_buffer(loader, pool, &buffer)?;
        staging.destroy(loader);

        Ok(buffer)
    }

    /// DOES NOT DO ANY BOUNDS CHECKING
    pub unsafe fn copy_data<T: Copy>(&self, loader: &Loader, data: &[T]) {
        get_align(loader, self).copy_from_slice(data);
    }
}

unsafe fn get_align<T: Copy>(loader: &Loader, buffer: &Buffer) -> Align<T> {
    let ptr = match loader.allocator.get_mapped_ptr(buffer.allocation) {
        Ok(mut ptr) => ptr.as_mut(),
        _ => loader
            .device
            .map_memory(
                loader.allocator.get_memory(buffer.allocation).unwrap(),
                0,
                buffer.size,
                vk::MemoryMapFlags::empty(),
            )
            .unwrap(),
    };

    Align::new(ptr, std::mem::align_of::<T>() as u64, buffer.size)
}

struct UniformBufferInternal<T> {
    buffer: Buffer,
    pointer: RefCell<Align<T>>,
}

impl<T> Destructible for UniformBufferInternal<T> {
    fn destroy(self, loader: &Loader) {
        if loader
            .allocator
            .get_mapped_ptr(self.buffer.allocation)
            .is_err()
        {
            unsafe {
                loader
                    .device
                    .unmap_memory(loader.allocator.get_memory(self.buffer.allocation).unwrap());
            }
        }

        self.buffer.destroy(loader);
    }
}

impl<T: Copy> UniformBufferInternal<T> {
    fn new(loader: &Loader, create_info: BufferCreateInfo) -> Result<Self> {
        let buffer = Buffer::new(loader, create_info)?;
        let pointer = unsafe { RefCell::new(get_align(loader, &buffer)) };

        Ok(Self { buffer, pointer })
    }
}

pub struct UniformBuffer<T: Copy> {
    buffers: ParitySet<UniformBufferInternal<T>>,
    value: Cell<T>,
}

impl<T: Copy> UniformBuffer<T> {
    pub fn new(
        loader: &Loader,
        description: &TypedResourceDescription<T>,
        value: T,
        name: Option<Identifier>,
    ) -> Result<Self> {
        let create_info = match description.deref() {
            ResourceDescription::Uniform(UniformDescription {
                stride,
                elements,
                host_visible,
                ..
            }) => BufferCreateInfo {
                size: stride * *elements as u64,
                name: name.unwrap_or_else(|| id!("Uniform Buffer")),
                usage: vk::BufferUsageFlags::UNIFORM_BUFFER,
                location: if *host_visible {
                    vk::MemoryLocation::CpuToGpu
                } else {
                    vk::MemoryLocation::GpuOnly
                },
            },
            _ => {
                return Err(anyhow!(
                    "Resource description [{:?}] not a uniform buffer description",
                    description.deref()
                ))
            }
        };

        Ok(Self {
            buffers: ParitySet::from_fn(|| UniformBufferInternal::new(loader, create_info.clone()))
                .into_iter()
                .collect::<Result<_>>()?,
            value: Cell::new(value),
        })
    }

    pub fn get_buffers(&self) -> ParitySet<&Buffer> {
        self.buffers.as_ref().ref_map(|v| &v.buffer)
    }

    pub fn copy(&self, parity: Parity, value: T) {
        self.value.set(value);
        self.buffers
            .get(parity)
            .pointer
            .borrow_mut()
            .copy_from_slice(&[value])
    }

    pub fn update(&self, parity: Parity, f: impl FnOnce(&mut T)) {
        let mut value = self.value.get();
        f(&mut value);
        self.copy(parity, value);
    }
}
