use crate::prelude::*;
use anyhow::Result;

pub struct Buffer {
    pub buffer: vk::Buffer,
    pub allocation: vk::Allocation,
}

pub struct BufferCreateInfo {
    pub size: vk::DeviceSize,
    pub name: Option<&'static str>,
    pub usage: vk::BufferUsageFlags,
    pub location: vk::MemoryLocation,
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

    Ok(Buffer { buffer, allocation })
}
