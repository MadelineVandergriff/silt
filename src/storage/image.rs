use std::ops::Add;

use crate::{loader::Loader, prelude::*};
use anyhow::Result;
use cached::proc_macro::once;
use itertools::Itertools;

#[derive(Debug, Clone)]
pub struct Image {
    pub image: vk::Image,
    pub view: vk::ImageView,
    pub allocation: vk::Allocation,
    pub size: vk::Extent3D,
}

pub struct ImageCreateInfo {
    pub width: u32,
    pub height: u32,
    pub mip_levels: u32,
    pub format: vk::Format,
    pub tiling: vk::ImageTiling,
    pub usage: vk::ImageUsageFlags,
    pub location: vk::MemoryLocation,
    pub samples: vk::SampleCountFlags,
    pub view_aspect: vk::ImageAspectFlags,
    pub name: Option<&'static str>
}

impl Default for ImageCreateInfo {
    fn default() -> Self {
        Self {
            width: 0,
            height: 0,
            mip_levels: 1,
            format: vk::Format::R8G8B8A8_SRGB,
            tiling: vk::ImageTiling::OPTIMAL,
            usage: vk::ImageUsageFlags::empty(),
            location: vk::MemoryLocation::GpuOnly,
            samples: vk::SampleCountFlags::from_raw(1),
            view_aspect: vk::ImageAspectFlags::COLOR,
            name: None
        }
    }
}

pub unsafe fn create_image(loader: &Loader, create_info: ImageCreateInfo) -> Result<Image> {
    let size = vk::Extent3D {
        width: create_info.width,
        height: create_info.height,
        depth: 1,
    };

    let texture_image_create_info = vk::ImageCreateInfo::builder()
        .image_type(vk::ImageType::TYPE_2D)
        .extent(size)
        .mip_levels(create_info.mip_levels)
        .array_layers(1)
        .format(create_info.format)
        .tiling(create_info.tiling)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .usage(create_info.usage)
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .samples(create_info.samples);

    let image = loader
        .device
        .create_image(&texture_image_create_info, None)?;
    let requirements = loader.device.get_image_memory_requirements(image);

    let allocation_create_info = vk::AllocationCreateInfo {
        name: create_info.name.unwrap_or("UNNAMED IMAGE"),
        requirements,
        location: create_info.location,
        linear: create_info.tiling == vk::ImageTiling::LINEAR,
        allocation_scheme: vk::AllocationScheme::GpuAllocatorManaged,
    };

    let allocation = loader.allocator.allocate(&allocation_create_info)?;
    loader
        .device
        .bind_image_memory(
            image,
            loader.allocator.get_memory(allocation)?,
            loader.allocator.get_offset(allocation)?,
        )
        .unwrap();

    let create_info = vk::ImageViewCreateInfo::builder()
        .image(image)
        .format(create_info.format)
        .view_type(vk::ImageViewType::TYPE_2D)
        .subresource_range(
            vk::ImageSubresourceRange::builder()
                .aspect_mask(create_info.view_aspect)
                .base_mip_level(0)
                .level_count(create_info.mip_levels)
                .base_array_layer(0)
                .layer_count(1)
                .build(),
        );

    let view = loader.device.create_image_view(&create_info, None)?;

    Ok(Image {
        image,
        view,
        allocation,
        size,
    })
}

pub unsafe fn find_supported_format(
    instance: &Instance,
    pdevice: vk::PhysicalDevice,
    candidates: impl IntoIterator<Item = vk::Format>,
    tiling: vk::ImageTiling,
    features: vk::FormatFeatureFlags,
) -> Option<vk::Format> {
    candidates.into_iter().find(|format| {
        let properties = instance.get_physical_device_format_properties(pdevice, *format);

        match tiling {
            vk::ImageTiling::LINEAR => properties.linear_tiling_features.contains(features),
            vk::ImageTiling::OPTIMAL => properties.optimal_tiling_features.contains(features),
            _ => false,
        }
    })
}

#[once]
pub unsafe fn get_depth_format(
    instance: &Instance,
    pdevice: vk::PhysicalDevice,
) -> Option<vk::Format> {
    unsafe { // Needed for the macro to compile
        find_supported_format(
            instance,
            pdevice,
            [
                vk::Format::D32_SFLOAT,
                vk::Format::D32_SFLOAT_S8_UINT,
                vk::Format::D24_UNORM_S8_UINT,
            ],
            vk::ImageTiling::OPTIMAL,
            vk::FormatFeatureFlags::DEPTH_STENCIL_ATTACHMENT,
        )
    }
}

pub unsafe fn get_surface_format(
    loader: &Loader,
    surface: vk::SurfaceKHR,
    pdevice: vk::PhysicalDevice,
) -> vk::SurfaceFormatKHR {
    loader
        .surface
        .get_physical_device_surface_formats(pdevice, surface)
        .unwrap()
        .into_iter()
        .find_or_first(|&format| format.format == vk::Format::B8G8R8A8_SRGB)
        .unwrap()
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Volume3D {
    pub extent: vk::Extent3D,
    pub offset: vk::Offset3D,
}

impl From<vk::Extent3D> for Volume3D {
    fn from(value: vk::Extent3D) -> Self {
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

    pub fn offset_by(self, offset: vk::Offset3D) -> Self {
        Self {
            offset: vk::Offset3D {
                x: self.offset.x + offset.x,
                y: self.offset.y + offset.y,
                z: self.offset.z + offset.z,
            },
            ..self
        }
    }
}