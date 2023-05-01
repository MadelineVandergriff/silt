use std::cell::{Cell, RefCell};

use itertools::Itertools;
use silt::macros::ShaderOptions;
use silt::model::{Vertex, MVP, Model};
use silt::pipeline::{FragmentShader, Shaders, VertexShader};
use silt::prelude::*;
use silt::properties::{DeviceFeatures, DeviceFeaturesRequest, ProvidedFeatures};
use silt::storage::buffer::get_bound_buffer;
use silt::storage::descriptors::{
    get_descriptors, BindingDescription, DescriptorFrequency, DescriptorWriter,
};
use silt::storage::image::{upload_texture, ImageFile, SampledImage};
use silt::sync::{get_command_pools, QueueRequest, QueueType, get_sync_primitives};
use silt::{bindable, shader};
use silt::{loader::*, swapchain::Swapchain};
use silt::{pipeline, storage::descriptors};

use anyhow::Result;

bindable!(
    Texture,
    vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
    DescriptorFrequency::Global,
    1
);

fn main() -> Result<()> {
    let loader_ci = LoaderCreateInfo {
        width: 1920,
        height: 1080,
        title: "Silt Example".into(),
        device_features: DeviceFeaturesRequest {
            required: DeviceFeatures::SAMPLER_ANISOTROPY,
            prefered: DeviceFeatures::IMAGE_CUBE_ARRAY | DeviceFeatures::SPARSE_BINDING,
        },
        queue_requests: vec![QueueRequest {
            ty: QueueType::Graphics,
            count: 1,
        }],
    };

    let (loader, handles) = Loader::new(loader_ci.clone()).unwrap();
    let LoaderHandles {
        debug_messenger,
        surface,
        pdevice,
        queues,
    } = handles;
    let features = ProvidedFeatures::new(&loader, pdevice);

    let present_pass = unsafe { pipeline::get_present_pass(&loader, pdevice, surface) };
    let swapchain = unsafe {
        RefCell::new(Swapchain::new(
            &loader,
            surface,
            pdevice,
            present_pass,
            loader_ci.width,
            loader_ci.height,
        )?)
    };
    let vertex = VertexShader::new::<Vertex, MVP>(
        &loader,
        shader!("../assets/shaders/model_loading.vert", ShaderOptions::HLSL)?,
    )?;
    let fragment = FragmentShader::new::<Texture>(
        &loader,
        shader!("../assets/shaders/model_loading.frag", ShaderOptions::HLSL)?,
    )?;
    let shaders = Shaders { vertex, fragment };
    let layouts = descriptors::get_layouts(&loader, &[&shaders.vertex, &shaders.fragment])?;
    let pipeline =
        unsafe { pipeline::get_pipeline(&loader, pdevice, present_pass, shaders, &layouts)? };
    let pools = get_command_pools(&loader, &queues, vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)?;

    let texture_image = ImageFile::new("assets/textures/viking_room.png".into())?;
    let texture =
        unsafe { upload_texture::<Texture>(&loader, features, &pools[0], texture_image)? };

    let model = Model::load(&loader, &pools[0], "assets/models/viking_room.obj".into())?;

    let (descriptor_pool, descriptors) = unsafe {
        get_descriptors(
            &loader,
            &layouts,
            &[model.mvp.get_write(), texture.get_write()],
        )?
    };

    let command_buffers = pools[0].get_main_command_buffers(&loader)?;
    let primitives = unsafe { get_sync_primitives(&loader) };

    let record_command_buffer = |parity: Parity, frame: usize| unsafe {
        let command_buffer = *command_buffers.get(parity);
        let swapchain = swapchain.borrow();
        let swap_frame = swapchain.frames.get(frame).unwrap();

        loader.device
            .begin_command_buffer(command_buffer, &vk::CommandBufferBeginInfo::default())
            .unwrap();

        let color_attachment_clear = vk::ClearValue {
            color: vk::ClearColorValue {
                float32: [0., 0., 0., 0.],
            },
        };

        let depth_attachment_clear = vk::ClearValue {
            depth_stencil: vk::ClearDepthStencilValue {
                depth: 1.,
                stencil: 0,
            },
        };

        let clear_values = [color_attachment_clear, depth_attachment_clear];

        let render_pass_info = vk::RenderPassBeginInfo::builder()
            .render_pass(present_pass)
            .framebuffer(swap_frame.framebuffer)
            .render_area(vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: swapchain.extent,
            })
            .clear_values(&clear_values);
        loader.device.cmd_begin_render_pass(
            command_buffer,
            &render_pass_info,
            vk::SubpassContents::INLINE,
        );

        loader.device.cmd_bind_pipeline(
            command_buffer,
            vk::PipelineBindPoint::GRAPHICS,
            pipeline,
        );

        loader.device
            .cmd_bind_vertex_buffers(command_buffer, 0, &[model.vertex_buffer.buffer], &[0]);
        loader.device.cmd_bind_index_buffer(
            command_buffer,
            model.index_buffer.buffer,
            0,
            vk::IndexType::UINT32,
        );

        let viewport = vk::Viewport {
            x: 0.,
            y: 0.,
            width: swapchain.extent.width as f32,
            height: swapchain.extent.height as f32,
            min_depth: 0.,
            max_depth: 1.,
        };
        loader.device
            .cmd_set_viewport(command_buffer, 0, std::slice::from_ref(&viewport));

        let scissor = vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent: swapchain.extent,
        };
        loader.device
            .cmd_set_scissor(command_buffer, 0, std::slice::from_ref(&scissor));

        loader.device.cmd_bind_descriptor_sets(
            command_buffer,
            vk::PipelineBindPoint::GRAPHICS,
            layouts.pipeline,
            0,
            &descriptors.values().flat_map(|d| d.iter()).map(|&d| d).collect_vec(),
            &[],
        );

        loader.device
            .cmd_draw_indexed(command_buffer, model.indices.len() as u32, 1, 0, 0, 0);
        loader.device.cmd_end_render_pass(command_buffer);
        loader.device
            .end_command_buffer(command_buffer)
            .unwrap();
    };

    std::thread::sleep(std::time::Duration::from_secs(1));
    Ok(())
}
