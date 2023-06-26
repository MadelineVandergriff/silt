use std::collections::HashMap;
use std::ops::Deref;

use anyhow::{anyhow, Result};
use itertools::{any, Itertools};

use crate::loader::Loader;
use crate::material::ShaderEffect;
use crate::prelude::*;
use crate::properties::get_sample_counts;
use crate::resources::ResourceDescription;
use crate::storage::image::{self, AttachmentReferenceType};

pub unsafe fn get_present_pass(
    loader: &Loader,
    pdevice: vk::PhysicalDevice,
    surface: vk::SurfaceKHR,
) -> vk::RenderPass {
    let surface_format = image::get_surface_format(loader, surface, pdevice);
    let msaa_samples = get_sample_counts(loader, pdevice);

    let color_attachment = vk::AttachmentDescription::builder()
        .format(surface_format.format)
        .samples(msaa_samples)
        .load_op(vk::AttachmentLoadOp::CLEAR)
        .store_op(vk::AttachmentStoreOp::STORE)
        .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
        .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .final_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);

    let depth_attachment = vk::AttachmentDescription::builder()
        .format(image::get_depth_format(&loader.instance, pdevice).unwrap())
        .samples(msaa_samples)
        .load_op(vk::AttachmentLoadOp::CLEAR)
        .store_op(vk::AttachmentStoreOp::STORE)
        .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
        .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .final_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);

    let color_resolve_attachment = vk::AttachmentDescription::builder()
        .format(surface_format.format)
        .samples(vk::SampleCountFlags::TYPE_1)
        .load_op(vk::AttachmentLoadOp::DONT_CARE)
        .store_op(vk::AttachmentStoreOp::STORE)
        .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
        .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .final_layout(vk::ImageLayout::PRESENT_SRC_KHR);

    let attachments = [
        color_attachment.build(),
        depth_attachment.build(),
        color_resolve_attachment.build(),
    ];

    let color_attachment_reference = vk::AttachmentReference {
        attachment: 0,
        layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
    };

    let depth_attachment_reference = vk::AttachmentReference {
        attachment: 1,
        layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
    };

    let color_resolve_attachment_reference = vk::AttachmentReference {
        attachment: 2,
        layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
    };

    let subpass = vk::SubpassDescription::builder()
        .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
        .color_attachments(std::slice::from_ref(&color_attachment_reference))
        .resolve_attachments(std::slice::from_ref(&color_resolve_attachment_reference))
        .depth_stencil_attachment(&depth_attachment_reference);

    let subpass_dependency = vk::SubpassDependency::builder()
        .src_subpass(vk::SUBPASS_EXTERNAL)
        .dst_subpass(0)
        .src_stage_mask(
            vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
        )
        .src_access_mask(vk::AccessFlags::empty())
        .dst_stage_mask(
            vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
        )
        .dst_access_mask(
            vk::AccessFlags::COLOR_ATTACHMENT_WRITE
                | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
        );

    let render_pass_create_info = vk::RenderPassCreateInfo::builder()
        .attachments(&attachments)
        .subpasses(std::slice::from_ref(&subpass))
        .dependencies(std::slice::from_ref(&subpass_dependency));

    let render_pass = loader
        .device
        .create_render_pass(&render_pass_create_info, None)
        .unwrap();

    render_pass
}

pub struct RenderPass {
    pub pass: vk::RenderPass,
    pub framebuffer: vk::Framebuffer,
}

pub fn build_render_pass<I, T>(loader: &Loader, resources: I) -> Result<vk::RenderPass>
where
    I: IntoIterator<Item = T>,
    T: Deref<Target = ResourceDescription>,
{
    let attachments = resources
        .into_iter()
        .filter_map(|resource| {
            match resource.deref() {
                ResourceDescription::Attachment(attachment) => Some(attachment.clone()),
                _ => None
            }
        })
        .map(|attachment| {
            vk::AttachmentDescription::builder()
                .final_layout(attachment.final_layout)
                .format(attachment.format)
                .load_op(load_op)
        })
}
