use anyhow::Result;
use std::ffi::CStr;

use crate::loader::get_msaa_samples;
use crate::storage::descriptors::Layouts;
use crate::{loader::Loader, prelude::*};

mod render_pass;
pub use render_pass::*;

mod shader;
pub use shader::*;

pub unsafe fn get_pipeline(
    loader: &Loader,
    pdevice: vk::PhysicalDevice,
    render_pass: vk::RenderPass,
    shaders: Shaders,
    layouts: &Layouts,
) -> Result<vk::Pipeline> {
    let msaa_samples = get_msaa_samples(loader, pdevice);

    let vertex_stage_create_info = vk::PipelineShaderStageCreateInfo::builder()
        .stage(shaders.vertex.shader_flags())
        .module(shaders.vertex.shader)
        .name(CStr::from_bytes_with_nul_unchecked(b"main\0"));

    let fragment_stage_create_info = vk::PipelineShaderStageCreateInfo::builder()
        .stage(shaders.fragment.shader_flags())
        .module(shaders.fragment.shader)
        .name(CStr::from_bytes_with_nul_unchecked(b"main\0"));

    let shader_stages = [*vertex_stage_create_info, *fragment_stage_create_info];

    let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
    let dynamic_state =
        vk::PipelineDynamicStateCreateInfo::builder().dynamic_states(&dynamic_states);

    let binding_descriptions = shaders.vertex.vertex_bindings();
    let attribute_descriptions = shaders.vertex.vertex_attributes();

    let vertex_input_state = vk::PipelineVertexInputStateCreateInfo::builder()
        .vertex_binding_descriptions(&binding_descriptions)
        .vertex_attribute_descriptions(&attribute_descriptions[..]);

    let input_assembly_state = vk::PipelineInputAssemblyStateCreateInfo::builder()
        .topology(vk::PrimitiveTopology::TRIANGLE_LIST)
        .primitive_restart_enable(false);

    let viewport_state = vk::PipelineViewportStateCreateInfo::builder()
        .viewport_count(1)
        .scissor_count(1);

    let rasterization_state = vk::PipelineRasterizationStateCreateInfo::builder()
        .depth_clamp_enable(false)
        .rasterizer_discard_enable(false)
        .polygon_mode(vk::PolygonMode::FILL)
        .line_width(1.)
        .cull_mode(vk::CullModeFlags::NONE)
        .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
        .depth_bias_enable(false);

    let multisample_state = vk::PipelineMultisampleStateCreateInfo::builder()
        .sample_shading_enable(false)
        .rasterization_samples(msaa_samples);

    let color_blend_attachment_state = vk::PipelineColorBlendAttachmentState::builder()
        .color_write_mask(vk::ColorComponentFlags::RGBA)
        .blend_enable(false);

    let color_blend_state = vk::PipelineColorBlendStateCreateInfo::builder()
        .logic_op_enable(false)
        .attachments(std::slice::from_ref(&color_blend_attachment_state));

    let depth_stencil_state = vk::PipelineDepthStencilStateCreateInfo::builder()
        .depth_test_enable(true)
        .depth_write_enable(true)
        .depth_compare_op(vk::CompareOp::LESS);

    let pipeline_create_info = vk::GraphicsPipelineCreateInfo::builder()
        .stages(&shader_stages)
        .vertex_input_state(&vertex_input_state)
        .input_assembly_state(&input_assembly_state)
        .viewport_state(&viewport_state)
        .rasterization_state(&rasterization_state)
        .multisample_state(&multisample_state)
        .color_blend_state(&color_blend_state)
        .dynamic_state(&dynamic_state)
        .depth_stencil_state(&depth_stencil_state)
        .layout(layouts.pipeline)
        .render_pass(render_pass)
        .subpass(0);

    let pipeline = loader
        .device
        .create_graphics_pipelines(
            vk::PipelineCache::null(),
            std::slice::from_ref(&pipeline_create_info),
            None,
        )
        .map_err(|e| e.1)?[0];

    Ok(pipeline)
}
