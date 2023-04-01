use std::{cell::RefCell, collections::HashMap, num::NonZeroU64, ptr::NonNull, ffi::c_void};

use crate::vk;
use anyhow::{anyhow, Result};
pub use ash::{Device, Entry, Instance};
pub use ash::extensions::{ext::DebugUtils, khr::{Surface, Swapchain}};
use uuid::Uuid;

/// Subset of gpu_allocator::vulkan::Allocator with managed allocation handles
pub struct Allocator {
    inner: RefCell<gpu_allocator::vulkan::Allocator>,
    allocations: RefCell<HashMap<vk::Allocation, gpu_allocator::vulkan::Allocation>>,
}

impl Allocator {
    fn with<T>(
        &self,
        allocation: vk::Allocation,
        f: impl FnOnce(&gpu_allocator::vulkan::Allocation) -> T,
    ) -> Result<T> {
        Ok(f(self.allocations.borrow().get(&allocation).ok_or(
            anyhow!(
                "Allocation {} not found, possible use after free",
                Uuid::from(allocation).as_urn()
            ),
        )?))
    }

    fn with_result<T>(
        &self,
        allocation: vk::Allocation,
        f: impl FnOnce(&gpu_allocator::vulkan::Allocation) -> Result<T>,
    ) -> Result<T> {
        f(self.allocations.borrow().get(&allocation).ok_or(
            anyhow!(
                "Allocation {} not found, possible use after free",
                Uuid::from(allocation).as_urn()
            ),
        )?)
    }

    pub fn new(desc: &vk::AllocatorCreateInfo) -> Result<Self> {
        Ok(Self {
            inner: RefCell::new(gpu_allocator::vulkan::Allocator::new(desc)?),
            allocations: Default::default(),
        })
    }

    pub fn allocate(&self, desc: &vk::AllocationCreateInfo<'_>) -> Result<vk::Allocation> {
        let allocation = self.inner.borrow_mut().allocate(desc)?;
        let uuid = Uuid::new_v4();
        if self
            .allocations
            .borrow_mut()
            .insert(uuid.into(), allocation)
            .is_some()
        {
            return Err(anyhow!("Allocation hanlde {} not unique", uuid.as_urn()));
        }
        Ok(uuid.into())
    }

    pub fn free(&self, allocation: vk::Allocation) -> Result<()> {
        let allocation = self
            .allocations
            .borrow_mut()
            .remove(&allocation)
            .ok_or(anyhow!(
                "Could not find allocation {}, possible double free error",
                Uuid::from(allocation).as_urn()
            ))?;
        self.inner.borrow_mut().free(allocation)?;
        Ok(())
    }

    pub fn get_chunk_id(&self, allocation: vk::Allocation) -> Result<NonZeroU64> {
        self.with_result(allocation, |a| a.chunk_id().ok_or(anyhow!("could not obtain chunk id")))
    }

    pub fn get_memory(&self, allocation: vk::Allocation) -> Result<vk::DeviceMemory> {
        self.with(allocation, |a| unsafe { a.memory() })
    }

    pub fn get_is_dedicated(&self, allocation: vk::Allocation) -> Result<bool> {
        self.with(allocation, gpu_allocator::vulkan::Allocation::is_dedicated)
    }

    pub fn get_offset(&self, allocation: vk::Allocation) -> Result<u64> {
        self.with(allocation, gpu_allocator::vulkan::Allocation::offset)
    }

    pub fn get_size(&self, allocation: vk::Allocation) -> Result<u64> {
        self.with(allocation, gpu_allocator::vulkan::Allocation::size)
    }

    pub fn get_mapped_ptr(&self, allocation: vk::Allocation) -> Result<NonNull<c_void>> {
        self.with_result(allocation, |a| a.mapped_ptr().ok_or(anyhow!("memory not host visible")))
    }

    pub fn get_is_null(&self, allocation: vk::Allocation) -> Result<bool> {
        self.with(allocation, gpu_allocator::vulkan::Allocation::is_null)
    }
}
