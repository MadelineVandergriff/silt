use std::cell::{Cell, RefCell};
use std::collections::HashMap;

use itertools::Itertools;
use silt::material::ShaderOptions;
use silt::model::{Model, Vertex, MVP};
use silt::pipeline::{FragmentShader, Shaders, VertexShader};
use silt::prelude::*;
use silt::properties::{DeviceFeatures, DeviceFeaturesRequest, ProvidedFeatures};
use silt::storage::buffer::get_bound_buffer;
use silt::storage::descriptors::{
    get_descriptors, BindingDescription, DescriptorFrequency, DescriptorWriter, Layouts,
};
use silt::storage::image::{upload_texture, ImageFile, SampledImage};
use silt::sync::{
    get_command_pools, get_sync_primitives, QueueHandle, QueueRequest, QueueType, SyncPrimitives, Recordable,
};
use silt::{bindable, shader};
use silt::{loader::*, swapchain::Swapchain};
use silt::{pipeline, storage::descriptors};

use anyhow::Result;
use winit::event::{Event, WindowEvent};
use winit::event_loop::ControlFlow;

bindable!(
    Texture,
    vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
    DescriptorFrequency::Global,
    1
);

fn main() -> Result<()> {
    let loader_ci = LoaderCreateInfo {
        width: 640,
        height: 480,
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
        shader!("../assets/shaders/model_loading.vert", vec![], ShaderOptions::HLSL)?,
    )?;
    let fragment = FragmentShader::new::<Texture>(
        &loader,
        shader!("../assets/shaders/model_loading.frag", vec![], ShaderOptions::HLSL)?,
    )?;
    let shaders = Shaders { vertex, fragment };
    let layouts = descriptors::get_layouts(&loader, &[&shaders.vertex, &shaders.fragment])?;
    let pipeline =
        unsafe { pipeline::get_present_pipeline(&loader, pdevice, present_pass, shaders, &layouts)? };
    let pools = get_command_pools(
        &loader,
        &queues,
        vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER,
    )?;

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

    let context = loader.context.take();
    let mut parity = Parity::Even;
    context.run(move |e, _, control_flow| {
        *control_flow = ControlFlow::Poll;
        match e {
            Event::MainEventsCleared => {
                parity.swap();

                model.mvp.update(&loader, parity, |mvp| {
                    *mvp = MVP {
                        projection: glam::Mat4::perspective_rh(
                            std::f32::consts::FRAC_PI_2,
                            aspect_ratio(&swapchain),
                            0.1,
                            1000.,
                        ),
                        ..Default::default()
                    };
                });

                draw_frame(
                    &loader,
                    parity,
                    &primitives,
                    &swapchain,
                    surface,
                    pdevice,
                    present_pass,
                    &command_buffers,
                    pipeline,
                    &model,
                    &layouts,
                    &descriptors,
                    &queues,
                );
            }
            Event::WindowEvent {
                event: WindowEvent::CloseRequested,
                ..
            } => *control_flow = ControlFlow::Exit,
            _ => {}
        }
    });

    std::thread::sleep(std::time::Duration::from_secs(1));
    Ok(())
}

fn record_command_buffer(
    loader: &Loader,
    parity: Parity,
    frame: usize,
    command_buffers: &ParitySet<vk::CommandBuffer>,
    swapchain: &RefCell<Swapchain>,
    present_pass: vk::RenderPass,
    pipeline: vk::Pipeline,
    model: &Model,
    layouts: &Layouts,
    descriptors: &HashMap<DescriptorFrequency, ParitySet<vk::DescriptorSet>>,
) {
    unsafe {
        let command_buffer = *command_buffers.get(parity);
        let swapchain = swapchain.borrow();
        let swap_frame = swapchain.frames.get(frame).unwrap();

        loader
            .device
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

        loader
            .device
            .cmd_bind_pipeline(command_buffer, vk::PipelineBindPoint::GRAPHICS, pipeline);

        let viewport = vk::Viewport {
            x: 0.,
            y: 0.,
            width: swapchain.extent.width as f32,
            height: swapchain.extent.height as f32,
            min_depth: 0.,
            max_depth: 1.,
        };
        loader
            .device
            .cmd_set_viewport(command_buffer, 0, std::slice::from_ref(&viewport));

        let scissor = vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent: swapchain.extent,
        };
        loader
            .device
            .cmd_set_scissor(command_buffer, 0, std::slice::from_ref(&scissor));

        loader.device.cmd_bind_descriptor_sets(
            command_buffer,
            vk::PipelineBindPoint::GRAPHICS,
            layouts.pipeline,
            0,
            &descriptors.values().map(|d| *d.get(parity)).collect_vec(),
            &[],
        );

        model.record(loader, command_buffer);
        loader.device.cmd_end_render_pass(command_buffer);
        loader.device.end_command_buffer(command_buffer).unwrap();
    }
}

fn recreate_swapchain(
    loader: &Loader,
    swapchain: &RefCell<Swapchain>,
    surface: vk::SurfaceKHR,
    pdevice: vk::PhysicalDevice,
    present_pass: vk::RenderPass,
) {
    unsafe { loader.device.device_wait_idle().unwrap() };
    swapchain.take().destroy(&loader);
    let size = loader.window.inner_size();
    swapchain.replace(unsafe {
        Swapchain::new(
            &loader,
            surface,
            pdevice,
            present_pass,
            size.width,
            size.height,
        )
        .unwrap()
    });
}

fn draw_frame(
    loader: &Loader,
    parity: Parity,
    primitives: &ParitySet<SyncPrimitives>,
    swapchain: &RefCell<Swapchain>,
    surface: vk::SurfaceKHR,
    pdevice: vk::PhysicalDevice,
    present_pass: vk::RenderPass,
    command_buffers: &ParitySet<vk::CommandBuffer>,
    pipeline: vk::Pipeline,
    model: &Model,
    layouts: &Layouts,
    descriptors: &HashMap<DescriptorFrequency, ParitySet<vk::DescriptorSet>>,
    queues: &Vec<QueueHandle>,
) {
    unsafe {
        let primitives = primitives.get(parity);

        loader
            .device
            .wait_for_fences(&[primitives.in_flight], true, u64::MAX)
            .unwrap();

        let (frame, swapchain_suboptimal) = loader
            .swapchain
            .acquire_next_image(
                swapchain.borrow().swapchain,
                u64::MAX,
                primitives.image_available,
                vk::Fence::null(),
            )
            .unwrap_or((0, true));

        if swapchain_suboptimal {
            println!("Swapchain OOD at Image Acquisition");
            recreate_swapchain(loader, swapchain, surface, pdevice, present_pass);
            return;
        }

        loader.device.reset_fences(&[primitives.in_flight]).unwrap();

        loader
            .device
            .reset_command_buffer(
                *command_buffers.get(parity),
                vk::CommandBufferResetFlags::empty(),
            )
            .unwrap();
        record_command_buffer(
            loader,
            parity,
            frame as usize,
            command_buffers,
            swapchain,
            present_pass,
            pipeline,
            model,
            layouts,
            descriptors,
        );

        loader
            .device
            .queue_submit(
                queues[0].queues[0],
                std::slice::from_ref(
                    &vk::SubmitInfo::builder()
                        .wait_semaphores(&[primitives.image_available])
                        .wait_dst_stage_mask(&[vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT])
                        .command_buffers(&[*command_buffers.get(parity)])
                        .signal_semaphores(&[primitives.render_finished]),
                ),
                primitives.in_flight,
            )
            .unwrap();

        let _borrow = swapchain.borrow();
        let present_info = vk::PresentInfoKHR::builder()
            .wait_semaphores(std::slice::from_ref(&primitives.render_finished))
            .swapchains(std::slice::from_ref(&_borrow.swapchain))
            .image_indices(std::slice::from_ref(&frame));
        let swapchain_suboptimal = loader
            .swapchain
            .queue_present(queues[0].queues[0], &present_info)
            .unwrap_or(true);

        if swapchain_suboptimal {
            println!("Swapchain OOD at Queue Presentation");
            recreate_swapchain(loader, swapchain, surface, pdevice, present_pass);
            return;
        }
    }
}

fn aspect_ratio(swapchain: &RefCell<Swapchain>) -> f32 {
    swapchain.borrow().extent.width as f32 / swapchain.borrow().extent.height as f32
}
