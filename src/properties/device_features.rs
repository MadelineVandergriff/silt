use std::ops::BitOr;

use bitflags::bitflags;
use crate::prelude::*;

bitflags! {
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    pub struct DeviceFeatures: u128 {
        const SAMPLER_ANISOTROPY    = 0b1 << 0;
        const SPARSE_BINDING        = 0b1 << 1;
        const IMAGE_CUBE_ARRAY      = 0b1 << 2;
    }
}

#[derive(Debug, Clone, Copy)]
pub struct DeviceFeaturesRequest {
    pub required: DeviceFeatures,
    pub prefered: DeviceFeatures,
}

#[derive(Debug, Clone, Copy)]
pub struct ProvidedFeatures {
    pub features: DeviceFeatures,
    pub pdevice: vk::PhysicalDevice,
    pub limits: vk::PhysicalDeviceLimits
}

impl ProvidedFeatures {
    pub fn sampler_anisotropy(&self) -> Option<f32> {
        if self.features.contains(DeviceFeatures::SAMPLER_ANISOTROPY) {
             Some(self.limits.max_sampler_anisotropy)
        } else {
            None
        }
    }

    pub fn new(loader: &Loader, pdevice: vk::PhysicalDevice) -> Self {
        let features: DeviceFeatures = unsafe { loader.instance.get_physical_device_features(pdevice).into() };
        let limits = unsafe { loader.instance.get_physical_device_properties(pdevice).limits };

        Self {
            features,
            pdevice,
            limits
        }
    }
}

impl Into<DeviceFeatures> for vk::PhysicalDeviceFeatures {
    fn into(self) -> DeviceFeatures {
        match self.sampler_anisotropy {
            vk::TRUE => DeviceFeatures::SAMPLER_ANISOTROPY,
            _ => DeviceFeatures::empty()
        }.bitor(match self.sparse_binding {
            vk::TRUE => DeviceFeatures::SPARSE_BINDING,
            _ => DeviceFeatures::empty()
        }).bitor(match self.image_cube_array {
            vk::TRUE => DeviceFeatures::IMAGE_CUBE_ARRAY,
            _ => DeviceFeatures::empty()
        })
    }
}

impl Into<vk::PhysicalDeviceFeatures> for DeviceFeatures {
    fn into(self) -> vk::PhysicalDeviceFeatures {
        vk::PhysicalDeviceFeatures::builder()
            .sampler_anisotropy(self.contains(DeviceFeatures::SAMPLER_ANISOTROPY))
            .sparse_binding(self.contains(DeviceFeatures::SPARSE_BINDING))
            .image_cube_array(self.contains(DeviceFeatures::IMAGE_CUBE_ARRAY))
            .build()
    }
}

