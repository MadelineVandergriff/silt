use super::{
    descriptors::{Bindable, DescriptorWrite, DescriptorWriter},
    image::Image,
};
use crate::{prelude::*, resources::{ParitySet, Parity}};
use crate::sync::CommandPool;
use anyhow::{anyhow, Result};
use std::cell::{Cell, RefCell};

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

#[derive(Debug, Clone, Copy)]
pub struct BufferCreateInfo {
    pub size: vk::DeviceSize,
    pub name: Option<&'static str>,
    pub usage: vk::BufferUsageFlags,
    pub location: vk::MemoryLocation,
}

pub struct BoundBuffer<T: Bindable> {
    pub buffers: ParitySet<Buffer>,
    pub mappings: Option<ParitySet<MemoryMapping<'static, T>>>,
    inner: Cell<T>,
}

impl<T: Bindable> Destructible for BoundBuffer<T> {
    fn destroy(self, loader: &Loader) {
        for buffer in self.buffers.into_iter() {
            buffer.destroy(loader);
        }
    }
}

impl<T: Bindable> BoundBuffer<T> {
    fn copy(&self, loader: &Loader, parity: Parity, value: &T) {
        if let Some(ref mapping) = self.mappings {
            mapping
                .get(parity)
                .copy_from_slice(std::slice::from_ref(value));
        } else {
            unsafe {
                map_buffer(loader, &self.buffers.get(parity))
                    .copy_from_slice(std::slice::from_ref(value))
            };
        }
    }

    pub fn update(&self, loader: &Loader, parity: Parity, f: impl FnOnce(&mut T)) {
        let mut value = self.inner.get();
        f(&mut value);
        self.copy(loader, parity, &value);
        self.inner.set(value);
    }
}

impl<T: Bindable + Default> DescriptorWriter for BoundBuffer<T> {
    fn get_write<'a>(&'a self) -> DescriptorWrite<'a> {
        DescriptorWrite::from_buffer(
            self.buffers.map(|buffer| {
                vk::DescriptorBufferInfo::builder()
                    .buffer(buffer.buffer)
                    .offset(0)
                    .range(buffer.size)
            }),
            T::default().binding(),
        )
    }
}

pub unsafe fn create_buffer(loader: &Loader, create_info: BufferCreateInfo) -> Result<Buffer> {
    let buffer_create_info = vk::BufferCreateInfo::builder()
        .size(create_info.size)
        .usage(create_info.usage)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);

    let buffer = loader.device.create_buffer(&buffer_create_info, None)?;
    let requirements = loader.device.get_buffer_memory_requirements(buffer);

    let allocation_create_info = vk::AllocationCreateInfo {
        name: create_info.name.unwrap_or("UNNAMED BUFFER"),
        requirements,
        location: create_info.location,
        linear: true,
        allocation_scheme: vk::AllocationScheme::GpuAllocatorManaged,
    };

    let allocation = loader.allocator.allocate(&allocation_create_info)?;
    loader.device.bind_buffer_memory(
        buffer,
        loader.allocator.get_memory(allocation)?,
        loader.allocator.get_offset(allocation)?,
    )?;

    Ok(Buffer {
        buffer,
        allocation,
        size: requirements.size,
    })
}

pub unsafe fn copy_buffer(
    loader: &Loader,
    pool: &CommandPool,
    src: &Buffer,
    dst: &Buffer,
    region: vk::BufferCopy,
) -> Result<()> {
    if region.size + region.src_offset > src.size {
        return Err(anyhow!(
            "region exceeded src bounds: [{:?}] [{:?}]",
            region,
            src
        ));
    }

    if region.size + region.dst_offset > dst.size {
        return Err(anyhow!(
            "region exceeded dst bounds: [{:?}] [{:?}]",
            region,
            dst
        ));
    }

    pool.execute_one_time_commands(loader, |loader, cmd| {
        loader
            .device
            .cmd_copy_buffer(cmd, src.buffer, dst.buffer, std::slice::from_ref(&region));
    })
}

pub unsafe fn copy_entire_buffer(
    loader: &Loader,
    pool: &CommandPool,
    src: &Buffer,
    dst: &Buffer,
) -> Result<()> {
    if src.size != dst.size {
        return Err(anyhow!("src and dst mismatch: [{:?}] [{:?}]", src, dst));
    }

    let region = vk::BufferCopy::builder().size(src.size).build();
    copy_buffer(loader, pool, src, dst, region)
}

pub unsafe fn copy_buffer_to_image(
    loader: &Loader,
    pool: &CommandPool,
    src: &Buffer,
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

    pool.execute_one_time_commands(loader, |loader, cmd| {
        loader.device.cmd_copy_buffer_to_image(
            cmd,
            src.buffer,
            dst.image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            std::slice::from_ref(&region),
        );
    })
}

pub unsafe fn copy_buffer_to_whole_image(
    loader: &Loader,
    pool: &CommandPool,
    src: &Buffer,
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

    copy_buffer_to_image(loader, pool, src, dst, *region)
}

pub struct MemoryMapping<'a, T: Copy> {
    loader: Option<&'a Loader>,
    allocation: vk::Allocation,
    align: RefCell<Align<T>>,
}

impl<'a, T: Copy> Drop for MemoryMapping<'a, T> {
    fn drop(&mut self) {
        unsafe {
            self.loader.map(|loader| {
                if loader.allocator.get_mapped_ptr(self.allocation).is_err() {
                    loader
                        .device
                        .unmap_memory(loader.allocator.get_memory(self.allocation).unwrap());
                }
            });
        }
    }
}

impl<'a, T: Copy> MemoryMapping<'a, T> {
    pub fn copy_from_slice(&self, slice: &[T]) {
        self.align.borrow_mut().copy_from_slice(slice);
    }

    fn new(loader: &'a Loader, allocation: vk::Allocation, align: Align<T>) -> Self {
        Self {
            loader: Some(loader),
            allocation,
            align: RefCell::new(align),
        }
    }

    fn new_persistent(allocation: vk::Allocation, align: Align<T>) -> MemoryMapping<'static, T> {
        MemoryMapping {
            loader: None,
            allocation,
            align: RefCell::new(align),
        }
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

pub unsafe fn map_buffer<'a, T: Copy>(loader: &'a Loader, buffer: &Buffer) -> MemoryMapping<'a, T> {
    MemoryMapping::new(loader, buffer.allocation, get_align(loader, buffer))
}

pub unsafe fn map_buffer_persistent<T: Copy>(
    loader: &Loader,
    buffer: &Buffer,
) -> MemoryMapping<'static, T> {
    MemoryMapping::new_persistent(buffer.allocation, get_align(loader, buffer))
}

pub fn upload_to_gpu<T: Copy>(
    loader: &Loader,
    pool: &CommandPool,
    data: &[T],
    usage: vk::BufferUsageFlags,
    name: Option<&'static str>,
) -> Result<Buffer> {
    let staging_ci = BufferCreateInfo {
        size: std::mem::size_of_val(data) as u64,
        name: None,
        usage: usage | vk::BufferUsageFlags::TRANSFER_SRC,
        location: vk::MemoryLocation::CpuToGpu,
    };

    let staging = unsafe { create_buffer(loader, staging_ci)? };
    unsafe { map_buffer(loader, &staging).copy_from_slice(data) };

    let buffer_ci = BufferCreateInfo {
        size: std::mem::size_of_val(data) as u64,
        name,
        usage: usage | vk::BufferUsageFlags::TRANSFER_DST,
        location: vk::MemoryLocation::GpuOnly,
    };

    let buffer = unsafe { create_buffer(loader, buffer_ci)? };

    unsafe { copy_entire_buffer(loader, pool, &staging, &buffer)? };
    staging.destroy(loader);

    Ok(buffer)
}

pub fn get_bound_buffer<T: Bindable + Copy>(
    loader: &Loader,
    usage: vk::BufferUsageFlags,
) -> Result<BoundBuffer<T>> {
    let buffer_ci = BufferCreateInfo {
        size: std::mem::size_of::<T>() as u64,
        name: None,
        usage,
        location: vk::MemoryLocation::CpuToGpu,
    };

    let buffers = ParitySet::from_fn(|| unsafe { create_buffer(loader, buffer_ci).unwrap() });
    let mappings = buffers.map(|buffer| unsafe { map_buffer_persistent(loader, buffer) });

    Ok(BoundBuffer {
        buffers,
        mappings: Some(mappings),
        inner: Cell::default(),
    })
}
