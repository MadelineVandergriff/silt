use crate::{prelude::*, properties::ProvidedFeatures, sync::CommandPool, id};
use anyhow::Result;
use cached::proc_macro::once;
use itertools::Itertools;
use std::cell::Cell;

use super::{Buffer, BufferCreateInfo};

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
    pub name: Identifier,
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
            name: NULL_ID.clone(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Layout {
    Initial,
    TransferSrc,
    TransferDst,
    FragmentRead,
    ColorAttachment,
    DepthAttachment,
    DepthStencilAttachment,
    Present,
}

impl Layout {
    pub fn get_layout(&self) -> vk::ImageLayout {
        match self {
            Layout::Initial => vk::ImageLayout::UNDEFINED,
            Layout::TransferSrc => vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            Layout::TransferDst => vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            Layout::FragmentRead => vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            Layout::ColorAttachment => vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            Layout::DepthAttachment => vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL,
            Layout::DepthStencilAttachment => vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
            Layout::Present => vk::ImageLayout::PRESENT_SRC_KHR,
        }
    }

    pub fn get_pipeline_stage(&self) -> vk::PipelineStageFlags {
        match self {
            Layout::Initial => vk::PipelineStageFlags::TOP_OF_PIPE,
            Layout::TransferSrc => vk::PipelineStageFlags::TRANSFER,
            Layout::TransferDst => vk::PipelineStageFlags::TRANSFER,
            Layout::FragmentRead => vk::PipelineStageFlags::FRAGMENT_SHADER,
            Layout::ColorAttachment => vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            Layout::DepthAttachment | Layout::DepthStencilAttachment => {
                vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS
                    | vk::PipelineStageFlags::LATE_FRAGMENT_TESTS
            }
            Layout::Present => vk::PipelineStageFlags::BOTTOM_OF_PIPE,
        }
    }

    pub fn get_access(&self) -> vk::AccessFlags {
        match self {
            Layout::Initial => vk::AccessFlags::NONE,
            Layout::TransferSrc => vk::AccessFlags::TRANSFER_READ,
            Layout::TransferDst => vk::AccessFlags::TRANSFER_WRITE,
            Layout::FragmentRead => vk::AccessFlags::SHADER_READ,
            Layout::ColorAttachment => {
                vk::AccessFlags::COLOR_ATTACHMENT_WRITE | vk::AccessFlags::COLOR_ATTACHMENT_READ
            }
            Layout::DepthAttachment | Layout::DepthStencilAttachment => {
                vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE
                    | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ
            }
            Layout::Present => vk::AccessFlags::SHADER_WRITE,
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
    pub layout: Cell<Layout>,
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
            name: create_info.name.as_str(),
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
            layout: Cell::new(Layout::Initial),
        })
    }

    pub fn transition_layout(
        &self,
        loader: &Loader,
        pool: &CommandPool,
        new_layout: Layout,
    ) -> Result<()> {
        let old_layout = self.layout.get();
        let old_stage = old_layout.get_pipeline_stage();
        let new_stage = new_layout.get_pipeline_stage();
        let src_access = old_layout.get_access();
        let dst_access = new_layout.get_access();

        self.layout.set(new_layout);

        pool.execute_one_time_commands(loader, |_, command_buffer| {
            let barrier = vk::ImageMemoryBarrier::builder()
                .old_layout(old_layout.get_layout())
                .new_layout(new_layout.get_layout())
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(self.image)
                .src_access_mask(src_access)
                .dst_access_mask(dst_access)
                .subresource_range(
                    vk::ImageSubresourceRange::builder()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .base_mip_level(0)
                        .level_count(self.mips)
                        .base_array_layer(0)
                        .layer_count(1)
                        .build(),
                );

            unsafe {
                loader.device.cmd_pipeline_barrier(
                    command_buffer,
                    old_stage,
                    new_stage,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    std::slice::from_ref(&barrier),
                )
            };
        })
    }

    pub fn generate_mipmaps(&self, loader: &Loader, pool: &CommandPool) {
        if self.layout.get() != Layout::TransferDst {
            self.transition_layout(loader, pool, Layout::TransferDst)
                .unwrap();
        }

        self.layout.set(Layout::FragmentRead);

        pool.execute_one_time_commands(loader, |_, command_buffer| {
            let mut barrier = vk::ImageMemoryBarrier::builder()
                .image(self.image)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .subresource_range(
                    vk::ImageSubresourceRange::builder()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .base_array_layer(0)
                        .layer_count(1)
                        .level_count(1)
                        .build(),
                )
                .build();

            for mip_level in 0..self.mips - 1 {
                let mip_width = (self.size.width >> mip_level).max(1);
                let mip_height = (self.size.height >> mip_level).max(1);

                barrier.subresource_range.base_mip_level = mip_level;
                barrier.old_layout = vk::ImageLayout::TRANSFER_DST_OPTIMAL;
                barrier.new_layout = vk::ImageLayout::TRANSFER_SRC_OPTIMAL;
                barrier.src_access_mask = vk::AccessFlags::TRANSFER_WRITE;
                barrier.dst_access_mask = vk::AccessFlags::TRANSFER_READ;

                unsafe {
                    loader.device.cmd_pipeline_barrier(
                        command_buffer,
                        vk::PipelineStageFlags::TRANSFER,
                        vk::PipelineStageFlags::TRANSFER,
                        vk::DependencyFlags::empty(),
                        &[],
                        &[],
                        &[barrier],
                    )
                };

                let blit = vk::ImageBlit::builder()
                    .src_offsets([
                        vk::Offset3D { x: 0, y: 0, z: 0 },
                        vk::Offset3D {
                            x: mip_width as i32,
                            y: mip_height as i32,
                            z: 1,
                        },
                    ])
                    .src_subresource(
                        vk::ImageSubresourceLayers::builder()
                            .base_array_layer(0)
                            .layer_count(1)
                            .mip_level(mip_level)
                            .aspect_mask(vk::ImageAspectFlags::COLOR)
                            .build(),
                    )
                    .dst_offsets([
                        vk::Offset3D { x: 0, y: 0, z: 0 },
                        vk::Offset3D {
                            x: (mip_width >> 1).max(1) as i32,
                            y: (mip_height >> 1).max(1) as i32,
                            z: 1,
                        },
                    ])
                    .dst_subresource(
                        vk::ImageSubresourceLayers::builder()
                            .base_array_layer(0)
                            .layer_count(1)
                            .mip_level(mip_level + 1)
                            .aspect_mask(vk::ImageAspectFlags::COLOR)
                            .build(),
                    );

                unsafe {
                    loader.device.cmd_blit_image(
                        command_buffer,
                        self.image,
                        vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                        self.image,
                        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                        std::slice::from_ref(&blit),
                        vk::Filter::LINEAR,
                    )
                };

                barrier.old_layout = vk::ImageLayout::TRANSFER_SRC_OPTIMAL;
                barrier.new_layout = vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL;
                barrier.src_access_mask = vk::AccessFlags::TRANSFER_READ;
                barrier.dst_access_mask = vk::AccessFlags::SHADER_READ;

                unsafe {
                    loader.device.cmd_pipeline_barrier(
                        command_buffer,
                        vk::PipelineStageFlags::TRANSFER,
                        vk::PipelineStageFlags::FRAGMENT_SHADER,
                        vk::DependencyFlags::empty(),
                        &[],
                        &[],
                        &[barrier],
                    )
                };
            }

            barrier.subresource_range.base_mip_level = self.mips - 1;
            barrier.old_layout = vk::ImageLayout::TRANSFER_DST_OPTIMAL;
            barrier.new_layout = vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL;
            barrier.src_access_mask = vk::AccessFlags::TRANSFER_WRITE;
            barrier.dst_access_mask = vk::AccessFlags::SHADER_READ;

            unsafe {
                loader.device.cmd_pipeline_barrier(
                    command_buffer,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::FRAGMENT_SHADER,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &[barrier],
                )
            };
        })
        .unwrap();
    }
}

#[derive(Debug, Clone)]
pub struct SwapchainImage {
    pub image: vk::Image,
    pub view: vk::ImageView,
    pub format: vk::Format,
    pub extent: vk::Extent3D,
    pub samples: vk::SampleCountFlags,
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

impl SampledImage {
    pub fn new(
        loader: &Loader,
        image: Image,
        features: ProvidedFeatures,
    ) -> Result<Self> {
        let create_info = vk::SamplerCreateInfo::builder()
            .mag_filter(vk::Filter::LINEAR)
            .min_filter(vk::Filter::LINEAR)
            .address_mode_u(vk::SamplerAddressMode::REPEAT)
            .address_mode_v(vk::SamplerAddressMode::REPEAT)
            .address_mode_w(vk::SamplerAddressMode::REPEAT)
            .anisotropy_enable(features.sampler_anisotropy().is_some())
            .max_anisotropy(features.sampler_anisotropy().unwrap_or_default())
            .border_color(vk::BorderColor::FLOAT_OPAQUE_WHITE)
            .compare_op(vk::CompareOp::ALWAYS)
            .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
            .min_lod(0.)
            .max_lod(image.mips as f32)
            .build();

        let sampler = unsafe { loader.device.create_sampler(&create_info, None)? };

        Ok(Self {
            image,
            sampler,
            properties: create_info,
        })
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

    pub fn upload_to_gpu(
        self,
        loader: &Loader,
        features: ProvidedFeatures,
        pool: &CommandPool,
    ) -> Result<SampledImage> {
        let buffer_ci = BufferCreateInfo {
            size: self.size,
            name: NULL_ID.clone(),
            usage: vk::BufferUsageFlags::TRANSFER_SRC,
            location: vk::MemoryLocation::CpuToGpu,
        };

        let src_buffer = Buffer::new(loader, buffer_ci)?;
        unsafe { src_buffer.copy_data(loader, &self.pixels) };

        let image_ci = ImageCreateInfo {
            width: self.width,
            height: self.height,
            mip_levels: self.max_mips,
            usage: vk::ImageUsageFlags::TRANSFER_DST
                | vk::ImageUsageFlags::TRANSFER_SRC
                | vk::ImageUsageFlags::SAMPLED,
            view_aspect: vk::ImageAspectFlags::COLOR,
            name: id!("Texture Image"),
            ..Default::default()
        };

        let image = Image::new(loader, image_ci)?;
        image.transition_layout(loader, pool, Layout::TransferDst)?;
        src_buffer.copy_to_entire_image(loader, pool, &image)?;
        image.generate_mipmaps(loader, pool);
        src_buffer.destroy(loader);

        SampledImage::new(loader, image, features)
    }
}

pub fn find_supported_format(
    instance: &Instance,
    pdevice: vk::PhysicalDevice,
    candidates: impl IntoIterator<Item = vk::Format>,
    tiling: vk::ImageTiling,
    features: vk::FormatFeatureFlags,
) -> Option<vk::Format> {
    candidates.into_iter().find(|format| {
        let properties =
            unsafe { instance.get_physical_device_format_properties(pdevice, *format) };

        match tiling {
            vk::ImageTiling::LINEAR => properties.linear_tiling_features.contains(features),
            vk::ImageTiling::OPTIMAL => properties.optimal_tiling_features.contains(features),
            _ => false,
        }
    })
}

#[once]
pub fn get_depth_format(instance: &Instance, pdevice: vk::PhysicalDevice) -> Option<vk::Format> {
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

pub fn get_surface_format(
    loader: &Loader,
    surface: vk::SurfaceKHR,
    pdevice: vk::PhysicalDevice,
) -> vk::SurfaceFormatKHR {
    unsafe {
        loader
            .surface
            .get_physical_device_surface_formats(pdevice, surface)
            .unwrap()
            .into_iter()
            .find_or_first(|&format| format.format == vk::Format::B8G8R8A8_SRGB)
            .unwrap()
    }
}

