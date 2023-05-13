use std::cell::Cell;

use crate::{loader::Loader, prelude::*, sync::CommandPool, properties::ProvidedFeatures};
use anyhow::Result;
use cached::proc_macro::once;
use itertools::Itertools;

use super::{buffer::*, descriptors::{DescriptorWriter, DescriptorWrite, BindingDescription, Bindable, ShaderBinding}};

#[derive(Debug, Clone, Default)]
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

pub struct SampledImage {
    pub image: Image,
    pub sampler: vk::Sampler,
    pub properties: vk::SamplerCreateInfo,
    pub binding: BindingDescription,
}

impl DescriptorWriter for SampledImage {
    fn get_write<'a>(&'a self) -> DescriptorWrite<'a> {
        DescriptorWrite::from_image(
            ParitySet::from_fn(|| {
                vk::DescriptorImageInfo::builder()
                    .image_layout(self.image.layout.get())
                    .image_view(self.image.view)
                    .sampler(self.sampler)
            }),
            self.binding
        )
    }
}

pub struct ImageFile {
    pub pixels: image::RgbaImage,
    pub width: u32,
    pub height: u32,
    pub size: u64,
    pub max_mips: u32,
}

impl ImageFile {
    pub fn new(path: std::path::PathBuf) -> Result<Self> {
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

pub unsafe fn create_image(loader: &Loader, create_info: ImageCreateInfo) -> Result<Image> {
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

    let image = loader.device.create_image(&image_ci, None)?;
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

    let view = loader.device.create_image_view(&view_ci, None)?;

    Ok(Image {
        image,
        view,
        allocation,
        size,
        mips: create_info.mip_levels,
        samples: create_info.samples,
        format: create_info.format,
        layout: Cell::new(vk::ImageLayout::UNDEFINED)
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
    unsafe {
        // Needed for the macro to compile
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

fn get_pipeline_stage(layout: vk::ImageLayout) -> Result<vk::PipelineStageFlags> {
    Ok(match layout {
        vk::ImageLayout::UNDEFINED => vk::PipelineStageFlags::TOP_OF_PIPE,
        vk::ImageLayout::TRANSFER_DST_OPTIMAL => vk::PipelineStageFlags::TRANSFER,
        vk::ImageLayout::TRANSFER_SRC_OPTIMAL => vk::PipelineStageFlags::TRANSFER,
        vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL => vk::PipelineStageFlags::FRAGMENT_SHADER,
        _ => return Err(anyhow::anyhow!("could not find proper pipeline stage")),
    })
}

fn get_access_flags(layout: vk::ImageLayout) -> Result<vk::AccessFlags> {
    Ok(match layout {
        vk::ImageLayout::UNDEFINED => vk::AccessFlags::empty(),
        vk::ImageLayout::TRANSFER_DST_OPTIMAL => vk::AccessFlags::TRANSFER_WRITE,
        vk::ImageLayout::TRANSFER_SRC_OPTIMAL => vk::AccessFlags::TRANSFER_READ,
        vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL => vk::AccessFlags::SHADER_READ,
        _ => return Err(anyhow::anyhow!("could not find proper pipeline stage")),
    })
}

pub unsafe fn transition_layout(loader: &Loader, pool: &CommandPool, image: &Image, new_layout: vk::ImageLayout) -> Result<()> {
    let old_layout = image.layout.get();
    let old_stage = get_pipeline_stage(old_layout)?;
    let new_stage = get_pipeline_stage(new_layout)?;
    let src_access = get_access_flags(old_layout)?;
    let dst_access = get_access_flags(new_layout)?;

    image.layout.set(new_layout);

    pool.execute_one_time_commands(loader, |_, command_buffer| {
        let barrier = vk::ImageMemoryBarrier::builder()
            .old_layout(old_layout)
            .new_layout(new_layout)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(image.image)
            .src_access_mask(src_access)
            .dst_access_mask(dst_access)
            .subresource_range(
                vk::ImageSubresourceRange::builder()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .base_mip_level(0)
                    .level_count(image.mips)
                    .base_array_layer(0)
                    .layer_count(1)
                    .build(),
            );

        loader.device.cmd_pipeline_barrier(
            command_buffer,
            old_stage,
            new_stage,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            std::slice::from_ref(&barrier),
        );
    })
} 

pub unsafe fn generate_mipmaps(loader: &Loader, pool: &CommandPool, image: &Image) {
    if image.layout.get() != vk::ImageLayout::TRANSFER_DST_OPTIMAL {
        transition_layout(loader, pool, image, vk::ImageLayout::TRANSFER_DST_OPTIMAL).unwrap();
    }

    image.layout.set(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);

    pool.execute_one_time_commands(loader, |_, command_buffer| {
        let mut barrier = vk::ImageMemoryBarrier::builder()
            .image(image.image)
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

        for mip_level in 0..image.mips - 1 {
            let mip_width = (image.size.width >> mip_level).max(1);
            let mip_height = (image.size.height >> mip_level).max(1);

            barrier.subresource_range.base_mip_level = mip_level;
            barrier.old_layout = vk::ImageLayout::TRANSFER_DST_OPTIMAL;
            barrier.new_layout = vk::ImageLayout::TRANSFER_SRC_OPTIMAL;
            barrier.src_access_mask = vk::AccessFlags::TRANSFER_WRITE;
            barrier.dst_access_mask = vk::AccessFlags::TRANSFER_READ;

            loader.device.cmd_pipeline_barrier(
                command_buffer,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[barrier],
            );

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

            loader.device.cmd_blit_image(
                command_buffer,
                image.image,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                image.image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                std::slice::from_ref(&blit),
                vk::Filter::LINEAR,
            );

            barrier.old_layout = vk::ImageLayout::TRANSFER_SRC_OPTIMAL;
            barrier.new_layout = vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL;
            barrier.src_access_mask = vk::AccessFlags::TRANSFER_READ;
            barrier.dst_access_mask = vk::AccessFlags::SHADER_READ;

            loader.device.cmd_pipeline_barrier(
                command_buffer,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::FRAGMENT_SHADER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[barrier],
            );
        }

        barrier.subresource_range.base_mip_level = image.mips - 1;
        barrier.old_layout = vk::ImageLayout::TRANSFER_DST_OPTIMAL;
        barrier.new_layout = vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL;
        barrier.src_access_mask = vk::AccessFlags::TRANSFER_WRITE;
        barrier.dst_access_mask = vk::AccessFlags::SHADER_READ;

        loader.device.cmd_pipeline_barrier(
            command_buffer,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::FRAGMENT_SHADER,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[barrier],
        );
    }).unwrap();
}

pub unsafe fn upload_texture<T: Default + Bindable>(loader: &Loader, features: ProvidedFeatures, pool: &CommandPool, file: ImageFile) -> Result<SampledImage> {
    let buffer_ci = BufferCreateInfo {
        size: file.size,
        name: None,
        usage: vk::BufferUsageFlags::TRANSFER_SRC,
        location: vk::MemoryLocation::CpuToGpu,
    };

    let src_buffer = create_buffer(loader, buffer_ci)?;
    map_buffer(loader, &src_buffer).copy_from_slice(&file.pixels);

    let image_ci = ImageCreateInfo {
        width: file.width,
        height: file.height,
        mip_levels: file.max_mips,
        usage: vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::SAMPLED,
        view_aspect: vk::ImageAspectFlags::COLOR,
        name: Some("Texture Image"),
        ..Default::default()
    };

    let image = create_image(loader, image_ci)?;
    transition_layout(loader, pool, &image, vk::ImageLayout::TRANSFER_DST_OPTIMAL)?;
    copy_buffer_to_whole_image(loader, pool, &src_buffer, &image)?;
    generate_mipmaps(loader, pool, &image);
    src_buffer.destroy(loader);

    get_sampler(loader, features, image, T::default().binding())
}

pub fn get_sampler(loader: &Loader, features: ProvidedFeatures, image: Image, binding: BindingDescription) -> Result<SampledImage> {
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
    
    Ok(SampledImage {
        image,
        sampler,
        properties: create_info,
        binding
    })
}

#[derive(Debug, Clone, Copy)]
pub enum AttachmentType {
    Color, Depth, Input(ShaderBinding), DepthInput(ShaderBinding), Resolve, DepthResolve, Swapchain
}

impl AttachmentType {
    pub fn get_usage(&self) -> vk::ImageUsageFlags {
        match self {
            AttachmentType::Color => vk::ImageUsageFlags::COLOR_ATTACHMENT,
            AttachmentType::Depth => vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
            AttachmentType::Input(_) => vk::ImageUsageFlags::INPUT_ATTACHMENT | vk::ImageUsageFlags::COLOR_ATTACHMENT,
            AttachmentType::DepthInput(_) => vk::ImageUsageFlags::INPUT_ATTACHMENT | vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
            AttachmentType::Resolve => vk::ImageUsageFlags::COLOR_ATTACHMENT,
            AttachmentType::DepthResolve => vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
            _ => panic!("Unsupported attachment type")
        }
    }

    pub fn get_aspect(&self) -> vk::ImageAspectFlags {
        match self {
            AttachmentType::Color => vk::ImageAspectFlags::COLOR,
            AttachmentType::Depth => vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL,
            AttachmentType::Input(_) => vk::ImageAspectFlags::COLOR,
            AttachmentType::DepthInput(_) => vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL,
            AttachmentType::Resolve => vk::ImageAspectFlags::COLOR,
            AttachmentType::DepthResolve => vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL,
            _ => panic!("Unsupported attachment type")
        }
    }

    pub fn get_layout(&self) -> vk::ImageLayout {
        match self {
            AttachmentType::Color => vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            AttachmentType::Depth => vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
            AttachmentType::Input(_) => vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            AttachmentType::DepthInput(_) => vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
            AttachmentType::Resolve => vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            AttachmentType::DepthResolve => vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
            AttachmentType::Swapchain => vk::ImageLayout::PRESENT_SRC_KHR,
            _ => panic!("Unsupported attachment type")
        }
    }

    pub fn get_load_op(&self) -> vk::AttachmentLoadOp {
        match self {
            AttachmentType::Resolve | AttachmentType::DepthResolve => vk::AttachmentLoadOp::DONT_CARE,
            _ => vk::AttachmentLoadOp::CLEAR,
        }
    }
}

pub fn create_framebuffer_attachment(loader: &Loader, extent: vk::Extent2D, ty: AttachmentType, format: vk::Format) -> Result<Image> {
    let image_ci = ImageCreateInfo {
        width: extent.width,
        height: extent.height,
        mip_levels: 1,
        format,
        tiling: vk::ImageTiling::OPTIMAL,
        usage: ty.get_usage(),
        location: vk::MemoryLocation::GpuOnly,
        samples: vk::SampleCountFlags::TYPE_1,
        view_aspect: ty.get_aspect(),
        name: Some("framebuffer attachment"),
    };

    unsafe { create_image(loader, image_ci) }
}