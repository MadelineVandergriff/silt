use std::collections::HashMap;

use anyhow::{anyhow, Result};
use itertools::{Itertools, any};

use crate::loader::Loader;
use crate::material::{CombinedResource, Resource, ResourceDescription, ShaderEffect};
use crate::prelude::*;
use crate::properties::get_sample_counts;
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

pub fn create_render_pass<'a>(
    loader: &Loader,
    resources: impl IntoIterator<Item = CombinedResource<'a>>,
) -> Result<RenderPass> {
    let (attachments, views, sizes, types): (Vec<_>, Vec<_>, Vec<_>, Vec<_>) = resources
        .into_iter()
        .filter_map(|resource| match (resource.resource, resource.description) {
            (Resource::Attachment(image), ResourceDescription::Attachment { ty, format }) => {
                Some((
                    vk::AttachmentDescription::builder()
                        .format(image.format)
                        .samples(image.samples)
                        .load_op(ty.get_load_op())
                        .store_op(vk::AttachmentStoreOp::STORE)
                        .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
                        .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
                        .initial_layout(vk::ImageLayout::UNDEFINED)
                        .final_layout(ty.get_layout())
                        .build(),
                    image.view,
                    image.size,
                    ty,
                ))
            }
            _ => None,
        })
        .multiunzip();

    let attachment_references = std::iter::zip(&attachments, &types)
        .enumerate()
        .group_by(|(_, (_, ty))| ty.get_reference_type())
        .into_iter()
        .map(|(ty, group)| {
            (
                ty,
                group
                    .map(|(idx, (attachment, _))| {
                        vk::AttachmentReference::builder()
                            .attachment(idx as u32)
                            .layout(attachment.final_layout)
                            .build()
                    })
                    .collect_vec(),
            )
        })
        .collect::<HashMap<_, _>>();

    let null = vec![];

    let mut subpass = vk::SubpassDescription::builder()
        .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
        .color_attachments(
            attachment_references
                .get(&AttachmentReferenceType::Color)
                .unwrap_or(&null),
        )
        .resolve_attachments(
            attachment_references
                .get(&AttachmentReferenceType::Resolve)
                .unwrap_or(&null),
        )
        .input_attachments(
            attachment_references
                .get(&AttachmentReferenceType::Input)
                .unwrap_or(&null),
        );

    if let Some(depth_stencil) = attachment_references.get(&AttachmentReferenceType::DepthStencil) {
        if depth_stencil.len() != 1 {
            return Err(anyhow!(
                "Subpass must have exactly 1 or 0 depth/stencil attachments"
            ));
        }

        subpass = subpass.depth_stencil_attachment(&depth_stencil[0]);
    }

    let start_dependency = vk::SubpassDependency::builder()
        .src_subpass(vk::SUBPASS_EXTERNAL)
        .dst_subpass(0)
        .src_stage_mask(vk::PipelineStageFlags::BOTTOM_OF_PIPE)
        .dst_stage_mask(
            vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
        )
        .src_access_mask(vk::AccessFlags::MEMORY_READ)
        .dst_access_mask(
            vk::AccessFlags::COLOR_ATTACHMENT_READ
                | vk::AccessFlags::COLOR_ATTACHMENT_WRITE
                | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ
                | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
        );

    let end_dependency = vk::SubpassDependency::builder()
        .src_subpass(0)
        .dst_subpass(vk::SUBPASS_EXTERNAL)
        .src_stage_mask(
            vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
        )
        .dst_stage_mask(vk::PipelineStageFlags::BOTTOM_OF_PIPE)
        .src_access_mask(
            vk::AccessFlags::COLOR_ATTACHMENT_READ
                | vk::AccessFlags::COLOR_ATTACHMENT_WRITE
                | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ
                | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
        )
        .dst_access_mask(vk::AccessFlags::MEMORY_READ);

    let dependencies = [start_dependency.build(), end_dependency.build()];

    let render_pass_ci = vk::RenderPassCreateInfo::builder()
        .attachments(&attachments)
        .dependencies(&dependencies)
        .subpasses(std::slice::from_ref(&subpass));

    let render_pass = unsafe { loader.device.create_render_pass(&render_pass_ci, None)? };

    if !sizes.iter().all_equal() {
        return Err(anyhow!("All framebuffer attachments must have the same size"))
    }

    let framebuffer_ci = vk::FramebufferCreateInfo::builder()
        .attachments(&views)
        .render_pass(render_pass)
        .layers(1)
        .width(sizes[0].width)
        .height(sizes[0].height);

    let framebuffer = unsafe { loader.device.create_framebuffer(&framebuffer_ci, None)? };

    Ok(RenderPass{
        pass: render_pass,
        framebuffer
    })
}
