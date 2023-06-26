use anyhow::{Result};
use itertools::{izip, Itertools};

use crate::loader::Loader;
use crate::prelude::*;
use crate::properties::get_sample_counts;
use crate::resources::{Image, get_surface_format, ImageCreateInfo, get_depth_format};

#[derive(Debug, Clone)]
pub struct SwapFrame {
    pub image: vk::Image,
    pub view: vk::ImageView,
    pub framebuffer: vk::Framebuffer,
}

#[derive(Debug, Clone)]
pub struct Swapchain {
    pub swapchain: vk::SwapchainKHR,
    pub extent: vk::Extent2D,
    pub depth: Image,
    pub color: Image,
    pub frames: Vec<SwapFrame>,
}

impl Swapchain {
    pub unsafe fn new(
        loader: &Loader,
        surface: vk::SurfaceKHR,
        pdevice: vk::PhysicalDevice,
        present_pass: vk::RenderPass,
        width: u32,
        height: u32,
    ) -> Result<Self> {
        let msaa_samples = get_sample_counts(loader, pdevice);
        let surface_capabilities = loader
            .surface
            .get_physical_device_surface_capabilities(pdevice, surface)?;
        let surface_format = get_surface_format(loader, surface, pdevice);
        let present_mode = loader
            .surface
            .get_physical_device_surface_present_modes(pdevice, surface)
            .unwrap()
            .into_iter()
            .find(|&mode| mode == vk::PresentModeKHR::MAILBOX)
            .unwrap_or(vk::PresentModeKHR::FIFO);

        let image_count = match surface_capabilities.max_image_count {
            0 => surface_capabilities.min_image_count + 1,
            max => (surface_capabilities.min_image_count + 1).min(max),
        };

        let extent = match surface_capabilities.current_extent.width {
            std::u32::MAX => vk::Extent2D { width, height },
            _ => surface_capabilities.current_extent,
        };

        let pre_transform = if surface_capabilities
            .supported_transforms
            .contains(vk::SurfaceTransformFlagsKHR::IDENTITY)
        {
            vk::SurfaceTransformFlagsKHR::IDENTITY
        } else {
            surface_capabilities.current_transform
        };

        let swapchain_create_info = vk::SwapchainCreateInfoKHR::builder()
            .surface(surface)
            .image_extent(extent)
            .min_image_count(image_count)
            .image_color_space(surface_format.color_space)
            .image_format(surface_format.format)
            .present_mode(present_mode)
            .pre_transform(pre_transform)
            .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
            .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .clipped(true)
            .image_array_layers(1);

        let swapchain = loader
            .swapchain
            .create_swapchain(&swapchain_create_info, None)?;

        let images = loader.swapchain.get_swapchain_images(swapchain)?;

        let image_views = images
            .iter()
            .map(|&img| {
                let image_view_create_info = vk::ImageViewCreateInfo::builder()
                    .format(surface_format.format)
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .components(vk::ComponentMapping {
                        r: vk::ComponentSwizzle::R,
                        g: vk::ComponentSwizzle::G,
                        b: vk::ComponentSwizzle::B,
                        a: vk::ComponentSwizzle::A,
                    })
                    .subresource_range(vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 1,
                    })
                    .image(img);
                loader
                    .device
                    .create_image_view(&image_view_create_info, None)
                    .unwrap()
            })
            .collect::<Vec<_>>();

        let color_image_ci = ImageCreateInfo {
            width: extent.width,
            height: extent.height,
            format: surface_format.format,
            samples: msaa_samples,
            tiling: vk::ImageTiling::OPTIMAL,
            usage: vk::ImageUsageFlags::TRANSIENT_ATTACHMENT
                | vk::ImageUsageFlags::COLOR_ATTACHMENT,
            ..Default::default()
        };

        let color = Image::new(loader, color_image_ci)?;

        let depth_format = get_depth_format(
            &loader.instance,
            pdevice
        )
        .unwrap();

        let depth_image_ci = ImageCreateInfo {
            width: extent.width,
            height: extent.height,
            format: depth_format,
            samples: msaa_samples,
            tiling: vk::ImageTiling::OPTIMAL,
            usage: vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
            view_aspect: vk::ImageAspectFlags::DEPTH,
            ..Default::default()
        };

        let depth = Image::new(loader, depth_image_ci)?;

        let framebuffers = image_views
            .iter()
            .map(|image_view| {
                let attachments = [color.view, depth.view, *image_view];

                let framebuffer_create_info = vk::FramebufferCreateInfo::builder()
                    .render_pass(present_pass)
                    .attachments(&attachments)
                    .width(extent.width)
                    .height(extent.height)
                    .layers(1);

                loader
                    .device
                    .create_framebuffer(&framebuffer_create_info, None)
                    .unwrap()
            })
            .collect_vec();

        let frames = izip!(images, image_views, framebuffers)
            .map(|(image, view, framebuffer)| SwapFrame {
                image,
                view,
                framebuffer,
            })
            .collect_vec();

        Ok(Self {
            swapchain,
            extent,
            depth,
            color,
            frames,
        })
    }

    pub unsafe fn cleanup(self, loader: &Loader) -> Result<()> {
        loader.device.device_wait_idle()?;
        self.destroy(loader);
        Ok(())
    }
}

impl Destructible for SwapFrame {
    fn destroy(self, loader: &Loader) {
        self.framebuffer.destroy(loader);
        self.view.destroy(loader);
    }
}

impl Destructible for Swapchain {
    fn destroy(self, loader: &Loader) {
        for frame in self.frames {
            frame.destroy(loader);
        }

        self.color.destroy(loader);
        self.depth.destroy(loader);
        self.swapchain.destroy(loader);
    }
}
