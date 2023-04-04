use crate::{prelude::*, loader::Loader};
use anyhow::Result;

pub struct Image {
    pub image: vk::Image,
    pub view: vk::ImageView,
    pub allocation: vk::Allocation,
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
        }
    }
}

pub unsafe fn create_image(
    loader: &Loader,
    image_ci: ImageCreateInfo,
) -> Result<Image> {
    let texture_image_create_info = vk::ImageCreateInfo::builder()
        .image_type(vk::ImageType::TYPE_2D)
        .extent(vk::Extent3D {
            width: image_ci.width,
            height: image_ci.height,
            depth: 1,
        })
        .mip_levels(image_ci.mip_levels)
        .array_layers(1)
        .format(image_ci.format)
        .tiling(image_ci.tiling)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .usage(image_ci.usage)
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .samples(image_ci.samples);

    let image = loader.device
        .create_image(&texture_image_create_info, None)?;
    let requirements = loader.device.get_image_memory_requirements(image);

    let allocation_create_info = vk::AllocationCreateInfo {
        name: "UNNAMED IMAGE",
        requirements,
        location: image_ci.location,
        linear: true,
        allocation_scheme: vk::AllocationScheme::GpuAllocatorManaged,
    };

    let allocation = loader.allocator.allocate(&allocation_create_info)?;
    loader.device
        .bind_image_memory(
            image,
            loader.allocator.get_memory(allocation)?,
            loader.allocator.get_offset(allocation)?,
        )
        .unwrap();

    let create_info = vk::ImageViewCreateInfo::builder()
        .image(image)
        .format(image_ci.format)
        .view_type(vk::ImageViewType::TYPE_2D)
        .subresource_range(
            vk::ImageSubresourceRange::builder()
                .aspect_mask(image_ci.view_aspect)
                .base_mip_level(0)
                .level_count(image_ci.mip_levels)
                .base_array_layer(0)
                .layer_count(1)
                .build(),
        );

    let view = loader.device.create_image_view(&create_info, None)?;

    Ok(Image {
        image,
        view,
        allocation,
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
