pub use ash::vk::*;
pub use gpu_allocator::vulkan::AllocationScheme;
pub use gpu_allocator::vulkan::AllocationCreateDesc as AllocationCreateInfo;
pub use gpu_allocator::vulkan::AllocatorCreateDesc as AllocatorCreateInfo;
pub use gpu_allocator::{AllocationError, AllocatorDebugSettings, MemoryLocation};

use uuid::Uuid;
use crate::prelude::Loader;

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

#[derive(Clone, Copy, Debug, Default)]
pub struct Volume3D {
    pub extent: Extent3D,
    pub offset: Offset3D,
}

impl From<Extent3D> for Volume3D {
    fn from(value: Extent3D) -> Self {
        Self {
            extent: value,
            ..Default::default()
        }
    }
}

impl Volume3D {
    pub fn min(&self) -> glam::IVec3 {
        glam::ivec3(self.offset.x, self.offset.y, self.offset.z)
    }

    pub fn max(&self) -> glam::IVec3 {
        glam::ivec3(
            self.offset.x + self.extent.width as i32, 
            self.offset.y + self.extent.height as i32, 
            self.offset.z + self.extent.depth as i32
        )
    }

    pub fn contains_pt(&self, other: glam::IVec3) -> bool {
        self.max().cmpge(other).all()
        && self.min().cmple(other).all()
    }

    pub fn contains(&self, other: &Self) -> bool {
        self.contains_pt(other.max())
        && self.contains_pt(other.min())
    }

    pub fn offset_by(self, offset: Offset3D) -> Self {
        Self {
            offset: Offset3D {
                x: self.offset.x + offset.x,
                y: self.offset.y + offset.y,
                z: self.offset.z + offset.z,
            },
            ..self
        }
    }
}