pub use ash::vk::*;
pub use gpu_allocator::vulkan::AllocationScheme;
pub use gpu_allocator::vulkan::AllocationCreateDesc as AllocationCreateInfo;
pub use gpu_allocator::vulkan::AllocatorCreateDesc as AllocatorCreateInfo;
pub use gpu_allocator::{AllocationError, AllocatorDebugSettings, MemoryLocation};

use uuid::Uuid;

/// Handle to gpu_allocator::vulkan::Allocation
#[derive(Clone, Copy, Default, Debug, PartialEq, Eq, Hash)]
pub struct Allocation(u128);

impl From<Uuid> for Allocation {
    fn from(uuid: Uuid) -> Self {
        Self(uuid.as_u128())
    }
}

impl From<Allocation> for Uuid {
    fn from(allocation: Allocation) -> Self {
        Self::from_u128(allocation.0)
    }
}