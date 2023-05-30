use crate::prelude::*;
use anyhow::Result;
use std::cell::Cell;

#[derive(Debug, Clone)]
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
    pub name: Option<&'static str>,
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
            name: None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Image {
    pub image: vk::Image,
    pub view: vk::ImageView,
    pub allocation: vk::Allocation,
    pub size: vk::Extent3D,
    pub mips: u32,
    pub samples: vk::SampleCountFlags,
    pub format: vk::Format,
    pub layout: Cell<vk::ImageLayout>,
}

impl Destructible for Image {
    fn destroy(self, loader: &Loader) {
        self.view.destroy(loader);
        self.image.destroy(loader);
        self.allocation.destroy(loader);
    }
}

impl Image {
    pub fn new(loader: &Loader, create_info: ImageCreateInfo) -> Result<Self> {
        let size = vk::Extent3D {
            width: create_info.width,
            height: create_info.height,
            depth: 1,
        };

        let image_ci = vk::ImageCreateInfo::builder()
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

        let image = unsafe { loader.device.create_image(&image_ci, None)? };
        let requirements = unsafe { loader.device.get_image_memory_requirements(image) };

        let allocation_create_info = vk::AllocationCreateInfo {
            name: create_info.name.unwrap_or("UNNAMED IMAGE"),
            requirements,
            location: create_info.location,
            linear: create_info.tiling == vk::ImageTiling::LINEAR,
            allocation_scheme: vk::AllocationScheme::GpuAllocatorManaged,
        };

        let allocation = loader.allocator.allocate(&allocation_create_info)?;
        unsafe {
            loader
                .device
                .bind_image_memory(
                    image,
                    loader.allocator.get_memory(allocation)?,
                    loader.allocator.get_offset(allocation)?,
                )
                .unwrap();
        }

        let view_ci = vk::ImageViewCreateInfo::builder()
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

        let view = unsafe { loader.device.create_image_view(&view_ci, None)? };

        Ok(Image {
            image,
            view,
            allocation,
            size,
            mips: create_info.mip_levels,
            samples: create_info.samples,
            format: create_info.format,
            layout: Cell::new(vk::ImageLayout::UNDEFINED),
        })
    }
}

#[derive(Debug, Clone)]
pub struct SampledImage {
    pub image: Image,
    pub sampler: vk::Sampler,
    pub properties: vk::SamplerCreateInfo,
}

impl Destructible for SampledImage {
    fn destroy(self, loader: &Loader) {
        self.sampler.destroy(loader);
        self.image.destroy(loader);
    }
}

#[derive(Debug, Clone)]
pub struct ImageFile {
    pub pixels: image::RgbaImage,
    pub width: u32,
    pub height: u32,
    pub size: u64,
    pub max_mips: u32,
}

impl ImageFile {
    pub fn new(path: impl AsRef<std::path::Path>) -> Result<Self> {
        let pixels = image::open(path)?.into_rgba8();
        let width = pixels.width();
        let height = pixels.height();
        let size = pixels.as_flat_samples().min_length().unwrap() as u64;
        let max_mips = (width.max(height) as f32).log2().floor() as u32 + 1;

        Ok(Self {
            pixels,
            width,
            height,
            size,
            max_mips,
        })
    }
}
