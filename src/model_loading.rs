use crate::prelude::*;
use crate::vk;
use ash::util::Align;
use itertools::Itertools;
use memoffset::offset_of;
use raw_window_handle::{HasRawDisplayHandle, HasRawWindowHandle};
use std::borrow::Cow;
use std::cell::RefCell;
use std::f32::consts::*;
use std::ffi::CStr;
use std::time::{Duration, Instant};
use winit::event::{Event, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::window::{Window, WindowBuilder};
use derive_more::*;
use crate::macros::ShaderOptions;
use crate::shader;

#[cfg(any(target_os = "macos", target_os = "ios"))]
use crate::vk::{
    KhrGetPhysicalDeviceProperties2Fn, KhrPortabilityEnumerationFn, KhrPortabilitySubsetFn,
};

#[derive(Clone, Copy, Debug, Default, From, Into)]
pub struct QueueFamilyIndex(u32);

#[derive(Copy, Clone, Debug, Default)]
pub struct Vertex {
    pub pos: glam::Vec3,
    pub color: glam::Vec3,
    pub tex_coord: glam::Vec2,
}

pub trait Bindable {
    fn get_binding_description() -> vk::VertexInputBindingDescription;
    fn get_attribute_descriptions() -> Vec<vk::VertexInputAttributeDescription>;
}

impl Bindable for Vertex {
    fn get_binding_description() -> vk::VertexInputBindingDescription {
        vk::VertexInputBindingDescription::builder()
            .binding(0)
            .stride(std::mem::size_of::<Self>() as u32)
            .input_rate(vk::VertexInputRate::VERTEX)
            .build()
    }

    fn get_attribute_descriptions() -> Vec<vk::VertexInputAttributeDescription> {
        vec![
            vk::VertexInputAttributeDescription::builder()
                .binding(0)
                .location(0)
                .format(vk::Format::R32G32B32_SFLOAT)
                .offset(offset_of!(Vertex, pos) as u32)
                .build(),
            vk::VertexInputAttributeDescription::builder()
                .binding(0)
                .location(1)
                .format(vk::Format::R32G32B32_SFLOAT)
                .offset(offset_of!(Vertex, color) as u32)
                .build(),
            vk::VertexInputAttributeDescription::builder()
                .binding(0)
                .location(2)
                .format(vk::Format::R32G32_SFLOAT)
                .offset(offset_of!(Vertex, tex_coord) as u32)
                .build(),
        ]
    }
}

#[derive(Clone, Copy)]
#[allow(dead_code)]
pub struct UniformBufferObject {
    model: glam::Mat4,
    view: glam::Mat4,
    projection: glam::Mat4,
}

#[derive(Clone, Copy)]
pub struct Image {
    pub image: vk::Image,
    pub mip_levels: u32,
    pub format: vk::Format,
}

pub struct FrameData {
    pub command_buffer: vk::CommandBuffer,
    pub uniform_buffer: vk::Buffer,
    pub uniform_buffer_allocation: vk::Allocation,
    pub uniform_buffer_mapping: RefCell<Align<UniformBufferObject>>,
    pub descriptor_set: vk::DescriptorSet,
    pub image_available: vk::Semaphore,
    pub render_finished: vk::Semaphore,
    pub in_flight: vk::Fence,
}

pub struct VulkanData {
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u32>,

    pub window: Window,
    pub event_loop: Option<EventLoop<()>>,

    pub entry: Entry,
    pub instance: Instance,

    pub debug_utils_loader: DebugUtils,
    pub debug_callback: vk::DebugUtilsMessengerEXT,

    pub surface_loader: Surface,
    pub surface: vk::SurfaceKHR,

    pub pdevice: vk::PhysicalDevice,
    pub device: Device,
    pub queue_family_index: QueueFamilyIndex,
    pub queue: vk::Queue,

    pub allocator: Allocator,

    pub surface_capabilities: vk::SurfaceCapabilitiesKHR,
    pub surface_format: vk::SurfaceFormatKHR,
    pub present_mode: vk::PresentModeKHR,
    pub msaa_samples: vk::SampleCountFlags,

    pub swapchain_loader: Option<Swapchain>,
    pub swapchain: Option<vk::SwapchainKHR>,
    pub image_extent: vk::Extent2D,

    pub images: Vec<vk::Image>,
    pub image_views: Vec<vk::ImageView>,

    pub color_image: vk::Image,
    pub color_allocation: Option<vk::Allocation>,
    pub color_view: vk::ImageView,

    pub depth_image: vk::Image,
    pub depth_allocation: Option<vk::Allocation>,
    pub depth_view: vk::ImageView,

    pub render_pass: vk::RenderPass,
    pub shader_modules: Vec<vk::ShaderModule>,
    pub pipeline_layout: vk::PipelineLayout,
    pub pipeline: vk::Pipeline,

    pub framebuffers: Vec<vk::Framebuffer>,

    pub vertex_buffer: vk::Buffer,
    pub vertex_buffer_allocation: vk::Allocation,
    pub index_buffer: vk::Buffer,
    pub index_buffer_allocation: vk::Allocation,

    pub texture_image: vk::Image,
    pub texture_allocation: vk::Allocation,
    pub texture_view: vk::ImageView,
    pub texture_sampler: vk::Sampler,

    pub command_pool: vk::CommandPool,
    pub transient_command_pool: vk::CommandPool,
    pub descriptor_pool: vk::DescriptorPool,

    pub frame_data: Vec<FrameData>,
    pub current_frame: usize,
}

impl VulkanData {
    pub fn new() -> Self {
        // SAFETY maintains all invariants required by the Vulkan spec
        unsafe {
            let frames_in_flight: u32 = 2;
            let (width, height) = (640, 480);
            let (vertices, indices) = load_model();
            let (window, event_loop) = get_window(width, height);
            let (entry, instance) = get_instance(&window);
            let (debug_utils_loader, debug_callback) = get_debug_hooks(&entry, &instance);
            let (surface_loader, surface) = get_surface(&window, &entry, &instance);
            let (pdevice, device, queue_family_index, queue, msaa_samples) =
                get_device_and_queue(&instance, &surface_loader, &surface);
            let allocator = get_allocator(&instance, &device, &pdevice);
            let (surface_capabilities, surface_format, present_mode) =
                get_surface_properties(&surface_loader, &surface, &pdevice);
            let (swapchain_loader, swapchain, image_extent) = get_swapchain(
                width,
                height,
                &instance,
                &device,
                &surface,
                surface_capabilities,
                surface_format,
                present_mode,
            );
            let (images, image_views) =
                get_image_views(&device, &swapchain_loader, &swapchain, surface_format);
            let (color_image, color_allocation, color_view) = get_color_resources(
                &device,
                &allocator,
                surface_format,
                msaa_samples,
                image_extent,
            );
            let (depth_image, depth_allocation, depth_view, depth_format) = get_depth_resources(
                &instance,
                &device,
                &pdevice,
                &allocator,
                msaa_samples,
                image_extent,
            );
            let render_pass = get_render_pass(&device, surface_format, msaa_samples, depth_format);
            let (shader_modules, descriptor_set_layout, pipeline_layout, pipeline) =
                get_pipeline(&device, &render_pass, msaa_samples);
            let framebuffers = get_framebuffers(
                &device,
                &image_views,
                image_extent,
                &color_view,
                &depth_view,
                &render_pass,
            );
            let (command_pool, transient_command_pool, command_buffers) =
                get_command_buffers(&device, queue_family_index, frames_in_flight);
            let (vertex_buffer, vertex_buffer_allocation) = get_vertex_buffer(
                &device,
                &queue,
                &transient_command_pool,
                &allocator,
                &vertices,
            );
            let (index_buffer, index_buffer_allocation) = get_index_buffer(
                &device,
                &queue,
                &transient_command_pool,
                &allocator,
                &indices,
            );
            let uniform_buffers =
                get_uniform_buffers(&device, &allocator, frames_in_flight);
            let (texture_image, texture_allocation, texture_view, texture_sampler) = get_texture(
                &instance,
                &pdevice,
                &device,
                &queue,
                &transient_command_pool,
                &allocator,
            );
            let (descriptor_pool, descriptor_sets) = get_descriptor_sets(
                &device,
                &descriptor_set_layout,
                &texture_view,
                &texture_sampler,
                &uniform_buffers,
            );
            let sync_objects = get_semaphores(&device, frames_in_flight);

            let frame_data = itertools::multizip((
                command_buffers,
                uniform_buffers,
                descriptor_sets,
                sync_objects,
            ))
            .map(
                |(
                    command_buffer,
                    (uniform_buffer, uniform_buffer_allocation, uniform_buffer_mapping),
                    descriptor_set,
                    (image_available, render_finished, in_flight),
                )| {
                    FrameData {
                        command_buffer,
                        uniform_buffer,
                        uniform_buffer_allocation,
                        uniform_buffer_mapping: RefCell::new(uniform_buffer_mapping),
                        descriptor_set,
                        image_available,
                        render_finished,
                        in_flight,
                    }
                },
            )
            .collect();

            Self {
                vertices,
                indices,
                window,
                event_loop: Some(event_loop),
                entry,
                instance,
                debug_utils_loader,
                debug_callback,
                surface_loader,
                surface,
                pdevice,
                device,
                queue_family_index,
                queue,
                allocator,
                surface_capabilities,
                surface_format,
                present_mode,
                msaa_samples,
                swapchain_loader: Some(swapchain_loader),
                swapchain: Some(swapchain),
                image_extent,
                images,
                image_views,
                color_image,
                color_allocation: Some(color_allocation),
                color_view,
                depth_image,
                depth_allocation: Some(depth_allocation),
                depth_view,
                render_pass,
                shader_modules,
                pipeline_layout,
                pipeline,
                framebuffers,
                vertex_buffer,
                vertex_buffer_allocation,
                index_buffer,
                index_buffer_allocation,
                texture_image,
                texture_allocation,
                texture_view,
                texture_sampler,
                command_pool,
                transient_command_pool,
                descriptor_pool,
                frame_data,
                current_frame: 0,
            }
        }
    }

    unsafe fn record_command_buffer(&self, image_index: u32, current_frame: usize) {
        let frame = &self.frame_data[current_frame];

        self.device
            .begin_command_buffer(frame.command_buffer, &vk::CommandBufferBeginInfo::default())
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
            .render_pass(self.render_pass)
            .framebuffer(*self.framebuffers.get(image_index as usize).unwrap())
            .render_area(vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: self.image_extent,
            })
            .clear_values(&clear_values);
        self.device.cmd_begin_render_pass(
            frame.command_buffer,
            &render_pass_info,
            vk::SubpassContents::INLINE,
        );

        self.device.cmd_bind_pipeline(
            frame.command_buffer,
            vk::PipelineBindPoint::GRAPHICS,
            self.pipeline,
        );

        self.device
            .cmd_bind_vertex_buffers(frame.command_buffer, 0, &[self.vertex_buffer], &[0]);
        self.device.cmd_bind_index_buffer(
            frame.command_buffer,
            self.index_buffer,
            0,
            vk::IndexType::UINT32,
        );

        let viewport = vk::Viewport {
            x: 0.,
            y: 0.,
            width: self.image_extent.width as f32,
            height: self.image_extent.height as f32,
            min_depth: 0.,
            max_depth: 1.,
        };
        self.device
            .cmd_set_viewport(frame.command_buffer, 0, std::slice::from_ref(&viewport));

        let scissor = vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent: self.image_extent,
        };
        self.device
            .cmd_set_scissor(frame.command_buffer, 0, std::slice::from_ref(&scissor));

        self.device.cmd_bind_descriptor_sets(
            frame.command_buffer,
            vk::PipelineBindPoint::GRAPHICS,
            self.pipeline_layout,
            0,
            &[frame.descriptor_set],
            &[],
        );

        self.device
            .cmd_draw_indexed(frame.command_buffer, self.indices.len() as u32, 1, 0, 0, 0);
        self.device.cmd_end_render_pass(frame.command_buffer);
        self.device
            .end_command_buffer(frame.command_buffer)
            .unwrap();
    }

    fn aspect_ratio(&self) -> f32 {
        self.image_extent.width as f32 / self.image_extent.height as f32
    }

    unsafe fn draw_frame(&mut self, current_frame: usize, spin_angle: f32, zoom: f32) {
        let frame = &self.frame_data[current_frame];

        self.device
            .wait_for_fences(&[frame.in_flight], true, u64::MAX)
            .unwrap();

        let (image_index, swapchain_suboptimal) = self
            .swapchain_loader
            .as_ref()
            .unwrap()
            .acquire_next_image(
                self.swapchain.unwrap(),
                u64::MAX,
                frame.image_available,
                vk::Fence::null(),
            )
            .unwrap_or((0, true));

        if swapchain_suboptimal {
            println!("Swapchain OOD at Image Acquisition");
            self.recreate_swapchain();
            return;
        }

        self.device.reset_fences(&[frame.in_flight]).unwrap();

        self.device
            .reset_command_buffer(frame.command_buffer, vk::CommandBufferResetFlags::empty())
            .unwrap();
        self.record_command_buffer(image_index, current_frame);

        let ubo = UniformBufferObject {
            model: glam::Mat4::from_rotation_z(spin_angle as f32),
            view: glam::Mat4::look_at_rh (
                glam::vec3(zoom, zoom, zoom),
                glam::vec3(0., 0., 0.),
                glam::vec3(0., 0., -1.),
            ),
            projection: glam::Mat4::perspective_rh(FRAC_PI_2, self.aspect_ratio(), 0.1, 100.),
        };
        frame
            .uniform_buffer_mapping
            .borrow_mut()
            .copy_from_slice(&[ubo]);

        self.device
            .queue_submit(
                self.queue,
                std::slice::from_ref(
                    &vk::SubmitInfo::builder()
                        .wait_semaphores(&[frame.image_available])
                        .wait_dst_stage_mask(&[vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT])
                        .command_buffers(&[frame.command_buffer])
                        .signal_semaphores(&[frame.render_finished]),
                ),
                frame.in_flight,
            )
            .unwrap();

        let present_info = vk::PresentInfoKHR::builder()
            .wait_semaphores(std::slice::from_ref(&frame.render_finished))
            .swapchains(std::slice::from_ref(self.swapchain.as_ref().unwrap()))
            .image_indices(std::slice::from_ref(&image_index));
        let swapchain_suboptimal = self
            .swapchain_loader
            .as_ref()
            .unwrap()
            .queue_present(self.queue, &present_info)
            .unwrap_or(true);

        if swapchain_suboptimal {
            println!("Swapchain OOD at Queue Presentation");
            self.recreate_swapchain();
            return;
        }
    }

    pub fn run(mut self) {
        let event_loop = self.event_loop.take().unwrap();
        let mut old_time = Instant::now();
        let mut frame_count: u64 = 0;
        let mut spin_angle: f32 = 0.;
        let mut zoom: f32 = 1.5;
        event_loop.run(move |event, _, control_flow| {
            *control_flow = ControlFlow::Poll;
            match event {
                Event::WindowEvent {
                    event: WindowEvent::CloseRequested,
                    ..
                } => *control_flow = ControlFlow::Exit,
                Event::DeviceEvent {
                    event: winit::event::DeviceEvent::MouseWheel { delta },
                    ..
                } => match delta {
                    winit::event::MouseScrollDelta::PixelDelta(pos) => {
                        if pos.x.abs() > 3. {
                            spin_angle -= pos.x as f32 / 70.;
                        }
                        if pos.y.abs() > 3. {
                            zoom += pos.y as f32 / 500.;
                            zoom = zoom.clamp(0.5, 8.);
                        }
                    }
                    _ => (),
                },
                Event::MainEventsCleared => unsafe {
                    if self.window.is_minimized().unwrap_or_default() {
                        std::thread::sleep(Duration::from_millis(100));
                    }
                    self.current_frame = (self.current_frame + 1) % self.frame_data.len();
                    let new_time = Instant::now();
                    let diff = new_time - old_time;
                    frame_count += 1;
                    if diff.as_secs() >= 1 {
                        println!("{:.2}FPS", frame_count as f32 / diff.as_secs_f32());
                        frame_count = 0;
                        old_time = new_time;
                    }
                    self.draw_frame(self.current_frame, spin_angle, zoom);
                },
                _ => (),
            }
        });
    }

    pub unsafe fn recreate_swapchain(&mut self) {
        self.device.device_wait_idle().unwrap();

        self.framebuffers.clear();
        self.image_views.clear();
        self.images.clear();
        self.swapchain_loader
            .as_ref()
            .unwrap()
            .destroy_swapchain(self.swapchain.unwrap(), None);
        self.swapchain = None;
        self.swapchain_loader = None;

        self.device.destroy_image_view(self.color_view, None);
        self.device.destroy_image(self.color_image, None);
        self.allocator
            .free(self.color_allocation.take().unwrap())
            .unwrap();

        self.device.destroy_image_view(self.depth_view, None);
        self.device.destroy_image(self.depth_image, None);
        self.allocator
            .free(self.depth_allocation.take().unwrap())
            .unwrap();

        while self.window.is_minimized().unwrap_or(false) {
            println!("Waiting for window visibility");
            std::thread::sleep(std::time::Duration::from_millis(50));
        }

        let size = self.window.inner_size();
        self.surface_capabilities =
            get_surface_properties(&self.surface_loader, &self.surface, &self.pdevice).0;

        let (swapchain_loader, swapchain, image_extent) = get_swapchain(
            size.width,
            size.height,
            &self.instance,
            &self.device,
            &self.surface,
            self.surface_capabilities,
            self.surface_format,
            self.present_mode,
        );

        let (images, image_views) = get_image_views(
            &self.device,
            &swapchain_loader,
            &swapchain,
            self.surface_format,
        );

        let (color_image, color_allocation, color_view) = get_color_resources(
            &self.device,
            &self.allocator,
            self.surface_format,
            self.msaa_samples,
            image_extent,
        );

        let (depth_image, depth_allocation, depth_view, _) = get_depth_resources(
            &self.instance,
            &self.device,
            &self.pdevice,
            &self.allocator,
            self.msaa_samples,
            image_extent,
        );

        let framebuffers = get_framebuffers(
            &self.device,
            &image_views,
            image_extent,
            &color_view,
            &depth_view,
            &self.render_pass,
        );

        self.swapchain_loader = Some(swapchain_loader);
        self.swapchain = Some(swapchain);
        self.image_extent = image_extent;
        self.images = images;
        self.image_views = image_views;
        self.framebuffers = framebuffers;
        self.color_image = color_image;
        self.color_allocation = Some(color_allocation);
        self.color_view = color_view;
        self.depth_image = depth_image;
        self.depth_allocation = Some(depth_allocation);
        self.depth_view = depth_view;
    }
}

impl Drop for VulkanData {
    fn drop(&mut self) {
        unsafe {
            self.device.queue_wait_idle(self.queue).unwrap();
            self.device
                .wait_for_fences(
                    &self
                        .frame_data
                        .iter()
                        .map(|d| d.in_flight)
                        .collect::<Vec<_>>()[..],
                    true,
                    u64::MAX,
                )
                .unwrap();
        };
    }
}

fn load_model() -> (Vec<Vertex>, Vec<u32>) {
    let (models, _) =
        tobj::load_obj("src/models/viking_room.obj", &tobj::GPU_LOAD_OPTIONS).unwrap();
    let mesh = &models[0].mesh;

    let vertices = std::iter::zip(
        mesh.positions.chunks_exact(3),
        mesh.texcoords.chunks_exact(2),
    )
    .map(|(pos, uv)| Vertex {
        pos: glam::vec3(pos[0], pos[1], pos[2]),
        color: glam::Vec3::ZERO,
        tex_coord: glam::vec2(uv[0], 1. - uv[1]),
    })
    .collect::<Vec<_>>();

    let indices = mesh.indices.clone();

    println!("Vertex count: [{}]", vertices.len());
    println!("Index count: [{}]", indices.len());

    (vertices, indices)
}

unsafe fn get_window(width: u32, height: u32) -> (Window, EventLoop<()>) {
    let icon = image::open("src/textures/icon.png").unwrap().into_rgba8();
    let (icon_width, icon_height) = icon.dimensions();
    let window_icon =
        winit::window::Icon::from_rgba(icon.into_raw(), icon_width, icon_height).unwrap();

    let event_loop = EventLoop::new();

    let window = WindowBuilder::new()
        .with_title("Silt Triangle")
        .with_inner_size(winit::dpi::LogicalSize::new(width, height))
        .with_window_icon(Some(window_icon))
        .build(&event_loop)
        .unwrap();

    (window, event_loop)
}

unsafe fn get_instance(window: &Window) -> (Entry, Instance) {
    let entry = Entry::linked();

    let layer_names =
        vec![CStr::from_bytes_with_nul_unchecked(b"VK_LAYER_KHRONOS_validation\0").as_ptr()];

    #[allow(unused_mut)]
    let mut portability_extensions: Vec<*const i8> = vec![];
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    {
        portability_extensions.push(KhrPortabilityEnumerationFn::name().as_ptr());
        portability_extensions.push(KhrGetPhysicalDeviceProperties2Fn::name().as_ptr());
    }

    let extension_names = ash_window::enumerate_required_extensions(window.raw_display_handle())
        .unwrap()
        .iter()
        .chain(std::iter::once(&DebugUtils::name().as_ptr()))
        .chain(&portability_extensions)
        .map(|ptr| *ptr)
        .collect::<Vec<_>>();

    let app_info = vk::ApplicationInfo::builder()
        .application_name(CStr::from_bytes_with_nul_unchecked(b"Silt Triangle\0"))
        .application_version(vk::make_api_version(0, 1, 0, 0))
        .engine_name(CStr::from_bytes_with_nul_unchecked(b"silt\0"))
        .engine_version(vk::make_api_version(0, 0, 1, 0))
        .api_version(vk::make_api_version(0, 1, 2, 0));

    let instance_flags = if cfg!(any(target_os = "macos", target_os = "ios")) {
        vk::InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR
    } else {
        vk::InstanceCreateFlags::empty()
    };

    let instance_create_info = vk::InstanceCreateInfo::builder()
        .flags(instance_flags)
        .application_info(&app_info)
        .enabled_layer_names(&layer_names)
        .enabled_extension_names(&extension_names);

    let instance = entry.create_instance(&instance_create_info, None).unwrap();

    (entry, instance)
}

unsafe fn get_surface(
    window: &Window,
    entry: &Entry,
    instance: &Instance,
) -> (Surface, vk::SurfaceKHR) {
    let surface_loader = Surface::new(entry, instance);

    let surface = ash_window::create_surface(
        entry,
        instance,
        window.raw_display_handle(),
        window.raw_window_handle(),
        None,
    )
    .unwrap();

    (surface_loader, surface)
}

unsafe fn get_debug_hooks(
    entry: &Entry,
    instance: &Instance,
) -> (DebugUtils, vk::DebugUtilsMessengerEXT) {
    let debug_info = vk::DebugUtilsMessengerCreateInfoEXT::builder()
        .message_severity(
            vk::DebugUtilsMessageSeverityFlagsEXT::ERROR
                | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
                | vk::DebugUtilsMessageSeverityFlagsEXT::INFO,
        )
        .message_type(
            vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
        )
        .pfn_user_callback(Some(vulkan_debug_callback));

    let debug_utils_loader = DebugUtils::new(entry, instance);
    let debug_callback = debug_utils_loader
        .create_debug_utils_messenger(&debug_info, None)
        .unwrap();

    (debug_utils_loader, debug_callback)
}

unsafe fn get_pdevice_suitability(
    instance: &Instance,
    surface_loader: &Surface,
    surface: &vk::SurfaceKHR,
    pdevice: &vk::PhysicalDevice,
) -> Option<(QueueFamilyIndex, vk::SampleCountFlags, i32)> {
    let queue_family_props = instance.get_physical_device_queue_family_properties(*pdevice);
    let properties = instance.get_physical_device_properties(*pdevice);
    let features = instance.get_physical_device_features(*pdevice);

    let queue_index = queue_family_props
        .into_iter()
        .enumerate()
        .find(|(idx, info)| {
            info.queue_flags.contains(vk::QueueFlags::GRAPHICS)
                && surface_loader
                    .get_physical_device_surface_support(*pdevice, *idx as u32, *surface)
                    .unwrap_or(false)
        })?
        .0 as u32;

    if features.sampler_anisotropy == vk::FALSE {
        return None;
    }

    let priority = match properties.device_type {
        vk::PhysicalDeviceType::DISCRETE_GPU => 0,
        vk::PhysicalDeviceType::INTEGRATED_GPU => 1,
        _ => i32::MAX,
    };

    let samples = properties.limits.framebuffer_color_sample_counts
        & properties.limits.framebuffer_depth_sample_counts;
    let samples = vk::SampleCountFlags::from_raw((samples.as_raw() + 1).next_power_of_two() >> 1);

    Some((queue_index.into(), samples, priority))
}

unsafe fn get_device_and_queue(
    instance: &Instance,
    surface_loader: &Surface,
    surface: &vk::SurfaceKHR,
) -> (
    vk::PhysicalDevice,
    Device,
    QueueFamilyIndex,
    vk::Queue,
    vk::SampleCountFlags,
) {
    let (pdevice, queue_family_index, msaa_samples, _) = instance
        .enumerate_physical_devices()
        .unwrap()
        .iter()
        .filter_map(|pdevice| {
            let (queue_index, msaa_samples, priority) =
                get_pdevice_suitability(instance, surface_loader, surface, pdevice)?;

            Some((*pdevice, queue_index, msaa_samples, priority))
        })
        .sorted_by_key(|(_, _, _, priority)| *priority)
        .next()
        .expect("Couldn't find suitable device.");

    let device_extensions_raw = [
        Swapchain::name().as_ptr(),
        #[cfg(any(target_os = "macos", target_os = "ios"))]
        KhrPortabilitySubsetFn::name().as_ptr(),
    ];

    let device_features = vk::PhysicalDeviceFeatures::builder().sampler_anisotropy(true);

    let queue_priorities = [1.0];
    let queue_create_info = vk::DeviceQueueCreateInfo::builder()
        .queue_family_index(queue_family_index.into())
        .queue_priorities(&queue_priorities);

    let device_create_info = vk::DeviceCreateInfo::builder()
        .enabled_extension_names(&device_extensions_raw)
        .enabled_features(&device_features)
        .queue_create_infos(std::slice::from_ref(&queue_create_info));

    let device = instance
        .create_device(pdevice, &device_create_info, None)
        .unwrap();
    let queue = device.get_device_queue(queue_family_index.into(), 0);

    (pdevice, device, queue_family_index, queue, msaa_samples)
}

unsafe fn get_allocator(
    instance: &Instance,
    device: &Device,
    pdevice: &vk::PhysicalDevice,
) -> Allocator {
    let allocator_create_info = vk::AllocatorCreateInfo {
        physical_device: *pdevice,
        device: device.clone(),
        instance: instance.clone(),
        debug_settings: Default::default(),
        buffer_device_address: false,
    };

    Allocator::new(&allocator_create_info).unwrap()
}

unsafe fn get_surface_properties(
    surface_loader: &Surface,
    surface: &vk::SurfaceKHR,
    pdevice: &vk::PhysicalDevice,
) -> (
    vk::SurfaceCapabilitiesKHR,
    vk::SurfaceFormatKHR,
    vk::PresentModeKHR,
) {
    let surface_capabilities = surface_loader
        .get_physical_device_surface_capabilities(*pdevice, *surface)
        .unwrap();
    let surface_format = surface_loader
        .get_physical_device_surface_formats(*pdevice, *surface)
        .unwrap()
        .into_iter()
        .find_or_first(|&format| format.format == vk::Format::B8G8R8A8_SRGB)
        .unwrap();
    let present_mode = surface_loader
        .get_physical_device_surface_present_modes(*pdevice, *surface)
        .unwrap()
        .into_iter()
        .find(|&mode| mode == vk::PresentModeKHR::MAILBOX)
        .unwrap_or(vk::PresentModeKHR::FIFO);

    (surface_capabilities, surface_format, present_mode)
}

unsafe fn find_supported_format(
    instance: &Instance,
    pdevice: &vk::PhysicalDevice,
    candidates: impl IntoIterator<Item = vk::Format>,
    tiling: vk::ImageTiling,
    features: vk::FormatFeatureFlags,
) -> Option<vk::Format> {
    candidates.into_iter().find(|format| {
        let properties = instance.get_physical_device_format_properties(*pdevice, *format);

        match tiling {
            vk::ImageTiling::LINEAR => properties.linear_tiling_features.contains(features),
            vk::ImageTiling::OPTIMAL => properties.optimal_tiling_features.contains(features),
            _ => false,
        }
    })
}

unsafe fn get_swapchain(
    width: u32,
    height: u32,
    instance: &Instance,
    device: &Device,
    surface: &vk::SurfaceKHR,
    surface_capabilities: vk::SurfaceCapabilitiesKHR,
    surface_format: vk::SurfaceFormatKHR,
    present_mode: vk::PresentModeKHR,
) -> (Swapchain, vk::SwapchainKHR, vk::Extent2D) {
    let image_count = match surface_capabilities.max_image_count {
        0 => surface_capabilities.min_image_count + 1,
        max => (surface_capabilities.min_image_count + 1).min(max),
    };

    let image_extent = match surface_capabilities.current_extent.width {
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

    let swapchain_loader = Swapchain::new(instance, device);

    let swapchain_create_info = vk::SwapchainCreateInfoKHR::builder()
        .surface(*surface)
        .image_extent(image_extent)
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

    let swapchain = swapchain_loader
        .create_swapchain(&swapchain_create_info, None)
        .unwrap();

    (swapchain_loader, swapchain, image_extent)
}

unsafe fn get_image_views(
    device: &Device,
    swapchain_loader: &Swapchain,
    swapchain: &vk::SwapchainKHR,
    surface_format: vk::SurfaceFormatKHR,
) -> (Vec<vk::Image>, Vec<vk::ImageView>) {
    let images = swapchain_loader.get_swapchain_images(*swapchain).unwrap();

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
            device
                .create_image_view(&image_view_create_info, None)
                .unwrap()
        })
        .collect::<Vec<_>>();

    (images, image_views)
}

unsafe fn get_shader_module(device: &Device, name: impl AsRef<str>) -> vk::ShaderModule {
    let code: Vec<u32> = shader!(name.as_ref(), ShaderOptions::CACHE).unwrap().into();
    let shader_module_create_info = vk::ShaderModuleCreateInfo::builder().code(&code[..]);

    device
        .create_shader_module(&shader_module_create_info, None)
        .unwrap()
}

unsafe fn get_render_pass(
    device: &Device,
    surface_format: vk::SurfaceFormatKHR,
    msaa_samples: vk::SampleCountFlags,
    depth_format: vk::Format,
) -> vk::RenderPass {
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
        .format(depth_format)
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

    let attachments = [color_attachment.build(), depth_attachment.build(), color_resolve_attachment.build()];

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

    let render_pass = device
        .create_render_pass(&render_pass_create_info, None)
        .unwrap();

    render_pass
}

unsafe fn get_pipeline(
    device: &Device,
    render_pass: &vk::RenderPass,
    msaa_samples: vk::SampleCountFlags,
) -> (
    Vec<vk::ShaderModule>,
    vk::DescriptorSetLayout,
    vk::PipelineLayout,
    vk::Pipeline,
) {
    let vertex_shader_module = get_shader_module(device, "src/shaders/texture_mapping.vert");
    let fragment_shader_module = get_shader_module(device, "src/shaders/texture_mapping.frag");

    let vertex_stage_create_info = vk::PipelineShaderStageCreateInfo::builder()
        .stage(vk::ShaderStageFlags::VERTEX)
        .module(vertex_shader_module)
        .name(CStr::from_bytes_with_nul_unchecked(b"main\0"));

    let fragment_stage_create_info = vk::PipelineShaderStageCreateInfo::builder()
        .stage(vk::ShaderStageFlags::FRAGMENT)
        .module(fragment_shader_module)
        .name(CStr::from_bytes_with_nul_unchecked(b"main\0"));

    let shader_stages = [*vertex_stage_create_info, *fragment_stage_create_info];

    let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
    let dynamic_state =
        vk::PipelineDynamicStateCreateInfo::builder().dynamic_states(&dynamic_states);

    let binding_descriptions = [Vertex::get_binding_description()];
    let attribute_descriptions = Vertex::get_attribute_descriptions();

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

    let ubo_binding = vk::DescriptorSetLayoutBinding::builder()
        .binding(0)
        .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
        .descriptor_count(1)
        .stage_flags(vk::ShaderStageFlags::VERTEX)
        .build();

    let sampler_binding = vk::DescriptorSetLayoutBinding::builder()
        .binding(1)
        .descriptor_count(1)
        .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
        .stage_flags(vk::ShaderStageFlags::FRAGMENT)
        .build();

    let descriptor_set_layout_bindings = [ubo_binding, sampler_binding];

    let descriptor_set_layout_create_info =
        vk::DescriptorSetLayoutCreateInfo::builder().bindings(&descriptor_set_layout_bindings);

    let descriptor_set_layout = device
        .create_descriptor_set_layout(&descriptor_set_layout_create_info, None)
        .unwrap();

    let pipeline_layout_create_info = vk::PipelineLayoutCreateInfo::builder()
        .set_layouts(std::slice::from_ref(&descriptor_set_layout));

    let pipeline_layout = device
        .create_pipeline_layout(&pipeline_layout_create_info, None)
        .unwrap();

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
        .layout(pipeline_layout)
        .render_pass(*render_pass)
        .subpass(0);

    let pipeline = device
        .create_graphics_pipelines(
            vk::PipelineCache::null(),
            std::slice::from_ref(&pipeline_create_info),
            None,
        )
        .unwrap()[0];

    (
        vec![vertex_shader_module, fragment_shader_module],
        descriptor_set_layout,
        pipeline_layout,
        pipeline,
    )
}

unsafe fn get_framebuffers(
    device: &Device,
    image_views: &Vec<vk::ImageView>,
    image_extent: vk::Extent2D,
    color_view: &vk::ImageView,
    depth_view: &vk::ImageView,
    render_pass: &vk::RenderPass,
) -> Vec<vk::Framebuffer> {
    image_views
        .iter()
        .map(|image_view| {
            let attachments = [*color_view, *depth_view, *image_view];

            let framebuffer_create_info = vk::FramebufferCreateInfo::builder()
                .render_pass(*render_pass)
                .attachments(&attachments)
                .width(image_extent.width)
                .height(image_extent.height)
                .layers(1);

            device
                .create_framebuffer(&framebuffer_create_info, None)
                .unwrap()
        })
        .collect()
}

unsafe fn create_buffer(
    device: &Device,
    allocator: &Allocator,
    size: u64,
    usage: vk::BufferUsageFlags,
    location: vk::MemoryLocation,
) -> (vk::Buffer, vk::Allocation, vk::MemoryRequirements) {
    let buffer_create_info = vk::BufferCreateInfo::builder()
        .size(size)
        .usage(usage)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);

    let buffer = device.create_buffer(&buffer_create_info, None).unwrap();
    let requirements = device.get_buffer_memory_requirements(buffer);

    let allocation_create_info = vk::AllocationCreateInfo {
        name: "UNNAMED BUFFER",
        requirements,
        location,
        linear: true,
        allocation_scheme: vk::AllocationScheme::GpuAllocatorManaged,
    };

    let allocation = allocator.allocate(&allocation_create_info).unwrap();
    device
        .bind_buffer_memory(buffer, allocator.get_memory(allocation).unwrap(), allocator.get_offset(allocation).unwrap())
        .unwrap();
    (buffer, allocation, requirements)
}

struct ImageCreateInfo {
    pub width: u32,
    pub height: u32,
    pub format: vk::Format,
    pub tiling: vk::ImageTiling,
    pub usage: vk::ImageUsageFlags,
    pub location: vk::MemoryLocation,
    pub mip_levels: u32,
    pub samples: vk::SampleCountFlags,
}

impl Default for ImageCreateInfo {
    fn default() -> Self {
        Self {
            width: 0,
            height: 0,
            format: vk::Format::UNDEFINED,
            tiling: vk::ImageTiling::LINEAR,
            usage: vk::ImageUsageFlags::empty(),
            location: vk::MemoryLocation::Unknown,
            mip_levels: 1,
            samples: vk::SampleCountFlags::TYPE_1,
        }
    }
}

unsafe fn create_image(
    device: &Device,
    allocator: &Allocator,
    create_info: ImageCreateInfo,
) -> (vk::Image, vk::Allocation, vk::MemoryRequirements) {
    let texture_image_create_info = vk::ImageCreateInfo::builder()
        .image_type(vk::ImageType::TYPE_2D)
        .extent(vk::Extent3D {
            width: create_info.width,
            height: create_info.height,
            depth: 1,
        })
        .mip_levels(create_info.mip_levels)
        .array_layers(1)
        .format(create_info.format)
        .tiling(create_info.tiling)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .usage(create_info.usage)
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .samples(create_info.samples);

    let image = device
        .create_image(&texture_image_create_info, None)
        .unwrap();
    let requirements = device.get_image_memory_requirements(image);

    let allocation_create_info = vk::AllocationCreateInfo {
        name: "UNNAMED IMAGE",
        requirements,
        location: create_info.location,
        linear: true,
        allocation_scheme: vk::AllocationScheme::GpuAllocatorManaged,
    };

    let allocation = allocator.allocate(&allocation_create_info).unwrap();
    device
        .bind_image_memory(image, allocator.get_memory(allocation).unwrap(), allocator.get_offset(allocation).unwrap())
        .unwrap();

    (image, allocation, requirements)
}

unsafe fn execute_one_time_commands(
    device: &Device,
    queue: &vk::Queue,
    command_pool: &vk::CommandPool,
    f: impl FnOnce(vk::CommandBuffer),
) {
    let command_buffer_create_info = vk::CommandBufferAllocateInfo::builder()
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_pool(*command_pool)
        .command_buffer_count(1);

    let command_buffer = device
        .allocate_command_buffers(&command_buffer_create_info)
        .unwrap()[0];

    let begin_info =
        vk::CommandBufferBeginInfo::builder().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
    device
        .begin_command_buffer(command_buffer, &begin_info)
        .unwrap();

    f(command_buffer);

    device.end_command_buffer(command_buffer).unwrap();

    let submit_info =
        vk::SubmitInfo::builder().command_buffers(std::slice::from_ref(&command_buffer));

    device
        .queue_submit(
            *queue,
            std::slice::from_ref(&submit_info),
            vk::Fence::null(),
        )
        .unwrap();
    device.queue_wait_idle(*queue).unwrap();
    device.free_command_buffers(*command_pool, &[command_buffer]);
}

unsafe fn copy_buffer(
    device: &Device,
    queue: &vk::Queue,
    command_pool: &vk::CommandPool,
    src: vk::Buffer,
    dst: vk::Buffer,
    size: u64,
) {
    execute_one_time_commands(device, queue, command_pool, |command_buffer| {
        let region = vk::BufferCopy::builder().size(size).build();
        device.cmd_copy_buffer(command_buffer, src, dst, &[region]);
    });
}

/// Copys a buffer to an image using a one-time command buffer
///
/// # Safety
///
/// Destination image must be have image layout ```TRANSFER_DST_OPTIMAL```
///
/// [https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkImageLayout.html]
unsafe fn copy_buffer_to_image(
    device: &Device,
    queue: &vk::Queue,
    command_pool: &vk::CommandPool,
    src: vk::Buffer,
    dst: vk::Image,
    width: u32,
    height: u32,
) {
    execute_one_time_commands(device, queue, command_pool, |command_buffer| {
        let region = vk::BufferImageCopy::builder()
            .buffer_offset(0)
            .buffer_row_length(0)
            .buffer_image_height(0)
            .image_subresource(
                vk::ImageSubresourceLayers::builder()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .mip_level(0)
                    .base_array_layer(0)
                    .layer_count(1)
                    .build(),
            )
            .image_offset(vk::Offset3D::default())
            .image_extent(vk::Extent3D {
                width,
                height,
                depth: 1,
            });

        device.cmd_copy_buffer_to_image(
            command_buffer,
            src,
            dst,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            std::slice::from_ref(&region),
        );
    });
}

unsafe fn transition_image_layout(
    device: &Device,
    queue: &vk::Queue,
    command_pool: &vk::CommandPool,
    image: vk::Image,
    mip_levels: u32,
    old_layout: vk::ImageLayout,
    new_layout: vk::ImageLayout,
) {
    execute_one_time_commands(device, queue, command_pool, |command_buffer| {
        let (src_access, dst_access, src_stage, dst_stage) = match (old_layout, new_layout) {
            (vk::ImageLayout::UNDEFINED, vk::ImageLayout::TRANSFER_DST_OPTIMAL) => (
                vk::AccessFlags::empty(),
                vk::AccessFlags::TRANSFER_WRITE,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::TRANSFER,
            ),
            (vk::ImageLayout::TRANSFER_DST_OPTIMAL, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL) => (
                vk::AccessFlags::TRANSFER_WRITE,
                vk::AccessFlags::SHADER_READ,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::FRAGMENT_SHADER,
            ),
            _ => panic!("Unsupported image layout transition"),
        };

        let barrier = vk::ImageMemoryBarrier::builder()
            .old_layout(old_layout)
            .new_layout(new_layout)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(image)
            .src_access_mask(src_access)
            .dst_access_mask(dst_access)
            .subresource_range(
                vk::ImageSubresourceRange::builder()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .base_mip_level(0)
                    .level_count(mip_levels)
                    .base_array_layer(0)
                    .layer_count(1)
                    .build(),
            );

        device.cmd_pipeline_barrier(
            command_buffer,
            src_stage,
            dst_stage,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            std::slice::from_ref(&barrier),
        );
    });
}

unsafe fn map_to_buffer<T: Copy>(
    device: &Device,
    allocator: &Allocator,
    allocation: vk::Allocation,
    requirements: vk::MemoryRequirements,
    slice: &[T],
) {
    let ptr = match allocator.get_mapped_ptr(allocation) {
        Ok(mut ptr) => ptr.as_mut(),
        _ => device
            .map_memory(
                allocator.get_memory(allocation).unwrap(),
                0,
                requirements.size,
                vk::MemoryMapFlags::empty(),
            )
            .unwrap(),
    };

    let mut align = Align::new(ptr, std::mem::align_of::<T>() as u64, requirements.size);
    align.copy_from_slice(slice);
    if allocator.get_mapped_ptr(allocation).is_err() {
        device.unmap_memory(allocator.get_memory(allocation).unwrap());
    }
}

unsafe fn persistent_map_to_buffer<T>(
    device: &Device,
    allocator: &Allocator,
    allocation: vk::Allocation,
    requirements: vk::MemoryRequirements,
) -> Align<T> {
    let ptr = match allocator.get_mapped_ptr(allocation) {
        Ok(mut ptr) => ptr.as_mut(),
        _ => device
            .map_memory(
                allocator.get_memory(allocation).unwrap(),
                0,
                requirements.size,
                vk::MemoryMapFlags::empty(),
            )
            .unwrap(),
    };
    Align::new(ptr, std::mem::align_of::<T>() as u64, requirements.size)
}

unsafe fn create_image_view(
    device: &Device,
    image: vk::Image,
    format: vk::Format,
    mip_levels: u32,
    aspect: vk::ImageAspectFlags,
) -> vk::ImageView {
    let create_info = vk::ImageViewCreateInfo::builder()
        .image(image)
        .format(format)
        .view_type(vk::ImageViewType::TYPE_2D)
        .subresource_range(
            vk::ImageSubresourceRange::builder()
                .aspect_mask(aspect)
                .base_mip_level(0)
                .level_count(mip_levels)
                .base_array_layer(0)
                .layer_count(1)
                .build(),
        );

    device.create_image_view(&create_info, None).unwrap()
}

unsafe fn generate_mipmaps(
    device: &Device,
    queue: &vk::Queue,
    command_pool: &vk::CommandPool,
    image: vk::Image,
    mip_levels: u32,
    width: u32,
    height: u32,
) {
    execute_one_time_commands(device, queue, command_pool, |command_buffer| {
        let mut barrier = vk::ImageMemoryBarrier::builder()
            .image(image)
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

        for mip_level in 0..mip_levels - 1 {
            let mip_width = (width >> mip_level).max(1);
            let mip_height = (height >> mip_level).max(1);

            barrier.subresource_range.base_mip_level = mip_level;
            barrier.old_layout = vk::ImageLayout::TRANSFER_DST_OPTIMAL;
            barrier.new_layout = vk::ImageLayout::TRANSFER_SRC_OPTIMAL;
            barrier.src_access_mask = vk::AccessFlags::TRANSFER_WRITE;
            barrier.dst_access_mask = vk::AccessFlags::TRANSFER_READ;

            device.cmd_pipeline_barrier(
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

            device.cmd_blit_image(
                command_buffer,
                image,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                std::slice::from_ref(&blit),
                vk::Filter::LINEAR,
            );

            barrier.old_layout = vk::ImageLayout::TRANSFER_SRC_OPTIMAL;
            barrier.new_layout = vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL;
            barrier.src_access_mask = vk::AccessFlags::TRANSFER_READ;
            barrier.dst_access_mask = vk::AccessFlags::SHADER_READ;

            device.cmd_pipeline_barrier(
                command_buffer,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::FRAGMENT_SHADER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[barrier],
            );
        }

        barrier.subresource_range.base_mip_level = mip_levels - 1;
        barrier.old_layout = vk::ImageLayout::TRANSFER_DST_OPTIMAL;
        barrier.new_layout = vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL;
        barrier.src_access_mask = vk::AccessFlags::TRANSFER_WRITE;
        barrier.dst_access_mask = vk::AccessFlags::SHADER_READ;

        device.cmd_pipeline_barrier(
            command_buffer,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::FRAGMENT_SHADER,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[barrier],
        );
    });
}

unsafe fn get_texture(
    instance: &Instance,
    pdevice: &vk::PhysicalDevice,
    device: &Device,
    queue: &vk::Queue,
    command_pool: &vk::CommandPool,
    allocator: &Allocator,
) -> (vk::Image, vk::Allocation, vk::ImageView, vk::Sampler) {
    let texture = image::open("src/textures/viking_room.png")
        .unwrap()
        .into_rgba8();
    let samples = texture.as_flat_samples();

    let (width, height) = texture.dimensions();
    let mip_levels = (width.max(height) as f32).log2().floor() as u32 + 1;
    let size = samples.min_length().unwrap() as u64;

    let (src_buffer, src_allocation, src_requirements) = create_buffer(
        device,
        allocator,
        size,
        vk::BufferUsageFlags::TRANSFER_SRC,
        vk::MemoryLocation::CpuToGpu,
    );
    map_to_buffer(
        device,
        allocator,
        src_allocation,
        src_requirements,
        samples.as_slice(),
    );

    let image_create_info = ImageCreateInfo {
        width,
        height,
        mip_levels,
        format: vk::Format::R8G8B8A8_SRGB,
        tiling: vk::ImageTiling::OPTIMAL,
        usage: vk::ImageUsageFlags::TRANSFER_DST
            | vk::ImageUsageFlags::TRANSFER_SRC
            | vk::ImageUsageFlags::SAMPLED,
        location: vk::MemoryLocation::GpuOnly,
        ..Default::default()
    };

    let (texture_image, texture_allocation, _) = create_image(device, allocator, image_create_info);

    transition_image_layout(
        device,
        queue,
        command_pool,
        texture_image,
        mip_levels,
        vk::ImageLayout::UNDEFINED,
        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
    );

    copy_buffer_to_image(
        device,
        queue,
        command_pool,
        src_buffer,
        texture_image,
        width,
        height,
    );

    generate_mipmaps(
        device,
        queue,
        command_pool,
        texture_image,
        mip_levels,
        width,
        height,
    );

    device.destroy_buffer(src_buffer, None);
    allocator.free(src_allocation).unwrap();

    let texture_view = create_image_view(
        device,
        texture_image,
        vk::Format::R8G8B8A8_SRGB,
        mip_levels,
        vk::ImageAspectFlags::COLOR,
    );

    let max_anisotropy = instance
        .get_physical_device_properties(*pdevice)
        .limits
        .max_sampler_anisotropy;

    let sampler_create_info = vk::SamplerCreateInfo::builder()
        .mag_filter(vk::Filter::LINEAR)
        .min_filter(vk::Filter::LINEAR)
        .address_mode_u(vk::SamplerAddressMode::REPEAT)
        .address_mode_v(vk::SamplerAddressMode::REPEAT)
        .address_mode_w(vk::SamplerAddressMode::REPEAT)
        .anisotropy_enable(true)
        .max_anisotropy(max_anisotropy)
        .border_color(vk::BorderColor::FLOAT_OPAQUE_WHITE)
        .compare_op(vk::CompareOp::ALWAYS)
        .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
        .min_lod(0.)
        .max_lod(mip_levels as f32);

    let texture_sampler = device.create_sampler(&sampler_create_info, None).unwrap();

    (
        texture_image,
        texture_allocation,
        texture_view,
        texture_sampler,
    )
}

unsafe fn get_vertex_buffer(
    device: &Device,
    queue: &vk::Queue,
    command_pool: &vk::CommandPool,
    allocator: &Allocator,
    vertices: &[Vertex],
) -> (vk::Buffer, vk::Allocation) {
    let (src_buffer, src_allocation, src_requirements) = create_buffer(
        device,
        allocator,
        std::mem::size_of_val(vertices) as u64,
        vk::BufferUsageFlags::TRANSFER_SRC,
        vk::MemoryLocation::CpuToGpu,
    );

    let (buffer, allocation, requirements) = create_buffer(
        device,
        allocator,
        std::mem::size_of_val(vertices) as u64,
        vk::BufferUsageFlags::VERTEX_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
        vk::MemoryLocation::GpuOnly,
    );

    map_to_buffer(device, allocator, src_allocation, src_requirements, vertices);
    copy_buffer(
        device,
        queue,
        command_pool,
        src_buffer,
        buffer,
        requirements.size,
    );

    device.destroy_buffer(src_buffer, None);
    allocator.free(src_allocation).unwrap();

    (buffer, allocation)
}

unsafe fn get_index_buffer(
    device: &Device,
    queue: &vk::Queue,
    command_pool: &vk::CommandPool,
    allocator: &Allocator,
    indices: &[u32],
) -> (vk::Buffer, vk::Allocation) {
    let (src_buffer, src_allocation, src_requirements) = create_buffer(
        device,
        allocator,
        std::mem::size_of_val(indices) as u64,
        vk::BufferUsageFlags::TRANSFER_SRC,
        vk::MemoryLocation::CpuToGpu,
    );

    let (buffer, allocation, requirements) = create_buffer(
        device,
        allocator,
        std::mem::size_of_val(indices) as u64,
        vk::BufferUsageFlags::INDEX_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
        vk::MemoryLocation::GpuOnly,
    );

    map_to_buffer(device, allocator, src_allocation, src_requirements, indices);
    copy_buffer(
        device,
        queue,
        command_pool,
        src_buffer,
        buffer,
        requirements.size,
    );

    device.destroy_buffer(src_buffer, None);
    allocator.free(src_allocation).unwrap();

    (buffer, allocation)
}

unsafe fn get_color_resources(
    device: &Device,
    allocator: &Allocator,
    surface_format: vk::SurfaceFormatKHR,
    msaa_samples: vk::SampleCountFlags,
    image_extent: vk::Extent2D,
) -> (vk::Image, vk::Allocation, vk::ImageView) {
    let color_format = surface_format.format;

    let image_create_info = ImageCreateInfo {
        width: image_extent.width,
        height: image_extent.height,
        format: color_format,
        samples: msaa_samples,
        tiling: vk::ImageTiling::OPTIMAL,
        usage: vk::ImageUsageFlags::TRANSIENT_ATTACHMENT | vk::ImageUsageFlags::COLOR_ATTACHMENT,
        location: vk::MemoryLocation::GpuOnly,
        ..Default::default()
    };

    let (color_image, color_allocation, _) = create_image(device, allocator, image_create_info);

    let color_view = create_image_view(
        device,
        color_image,
        color_format,
        1,
        vk::ImageAspectFlags::COLOR,
    );

    (color_image, color_allocation, color_view)
}

unsafe fn get_depth_resources(
    instance: &Instance,
    device: &Device,
    pdevice: &vk::PhysicalDevice,
    allocator: &Allocator,
    msaa_samples: vk::SampleCountFlags,
    image_extent: vk::Extent2D,
) -> (vk::Image, vk::Allocation, vk::ImageView, vk::Format) {
    let format = find_supported_format(
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
    .unwrap();

    let image_create_info = ImageCreateInfo {
        width: image_extent.width,
        height: image_extent.height,
        format,
        tiling: vk::ImageTiling::OPTIMAL,
        usage: vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
        location: vk::MemoryLocation::GpuOnly,
        samples: msaa_samples,
        ..Default::default()
    };

    let (depth_image, depth_allocation, _) = create_image(device, allocator, image_create_info);

    let depth_view = create_image_view(device, depth_image, format, 1, vk::ImageAspectFlags::DEPTH);

    (depth_image, depth_allocation, depth_view, format)
}

unsafe fn get_uniform_buffers(
    device: &Device,
    allocator: &Allocator,
    frames_in_flight: u32,
) -> Vec<(vk::Buffer, vk::Allocation, Align<UniformBufferObject>)> {
    (0..frames_in_flight)
        .map(|_| {
            let (buffer, allocation, requirements) = create_buffer(
                device,
                allocator,
                std::mem::size_of::<UniformBufferObject>() as u64,
                vk::BufferUsageFlags::UNIFORM_BUFFER,
                vk::MemoryLocation::CpuToGpu,
            );

            let persistent_mapping = persistent_map_to_buffer(device, allocator, allocation, requirements);

            (buffer, allocation, persistent_mapping)
        })
        .collect()
}

unsafe fn get_command_buffers(
    device: &Device,
    queue_family_index: QueueFamilyIndex,
    frames_in_flight: u32,
) -> (vk::CommandPool, vk::CommandPool, Vec<vk::CommandBuffer>) {
    let command_pool_create_info = vk::CommandPoolCreateInfo::builder()
        .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
        .queue_family_index(queue_family_index.into());

    let command_pool = device
        .create_command_pool(&command_pool_create_info, None)
        .unwrap();

    let transient_command_pool_create_info = vk::CommandPoolCreateInfo::builder()
        .flags(vk::CommandPoolCreateFlags::TRANSIENT)
        .queue_family_index(queue_family_index.into());

    let transient_command_pool = device
        .create_command_pool(&transient_command_pool_create_info, None)
        .unwrap();

    let command_buffer_alloc_info = vk::CommandBufferAllocateInfo::builder()
        .command_pool(command_pool)
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_buffer_count(frames_in_flight);

    let command_buffers = device
        .allocate_command_buffers(&command_buffer_alloc_info)
        .unwrap();

    (command_pool, transient_command_pool, command_buffers)
}

unsafe fn get_descriptor_sets(
    device: &Device,
    descriptor_set_layout: &vk::DescriptorSetLayout,
    texture_view: &vk::ImageView,
    texture_sampler: &vk::Sampler,
    uniform_buffers: &Vec<(vk::Buffer, vk::Allocation, Align<UniformBufferObject>)>,
) -> (vk::DescriptorPool, Vec<vk::DescriptorSet>) {
    let frames_in_flight = uniform_buffers.len() as u32;

    let pool_sizes = [
        vk::DescriptorPoolSize::builder()
            .ty(vk::DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(frames_in_flight)
            .build(),
        vk::DescriptorPoolSize::builder()
            .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(frames_in_flight)
            .build(),
    ];

    let descriptor_pool_create_info = vk::DescriptorPoolCreateInfo::builder()
        .pool_sizes(&pool_sizes)
        .max_sets(frames_in_flight);

    let descriptor_pool = device
        .create_descriptor_pool(&descriptor_pool_create_info, None)
        .unwrap();

    let layouts = vec![*descriptor_set_layout; frames_in_flight as usize];
    let descriptor_set_alloc_info = vk::DescriptorSetAllocateInfo::builder()
        .set_layouts(&layouts[..])
        .descriptor_pool(descriptor_pool);

    let descriptor_sets = device
        .allocate_descriptor_sets(&descriptor_set_alloc_info)
        .unwrap();
    assert!(descriptor_sets.len() == frames_in_flight as usize);

    for (descriptor_set, (buffer, _, _)) in descriptor_sets.iter().zip(uniform_buffers) {
        let buffer_info = vk::DescriptorBufferInfo::builder()
            .buffer(*buffer)
            .offset(0)
            .range(std::mem::size_of::<UniformBufferObject>() as u64);

        let uniform_write = vk::WriteDescriptorSet::builder()
            .buffer_info(std::slice::from_ref(&buffer_info))
            .dst_set(*descriptor_set)
            .dst_binding(0)
            .dst_array_element(0)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER);

        let image_info = vk::DescriptorImageInfo::builder()
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .image_view(*texture_view)
            .sampler(*texture_sampler);

        let sampler_write = vk::WriteDescriptorSet::builder()
            .image_info(std::slice::from_ref(&image_info))
            .dst_set(*descriptor_set)
            .dst_binding(1)
            .dst_array_element(0)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER);

        device.update_descriptor_sets(&[uniform_write.build(), sampler_write.build()], &[]);
    }

    (descriptor_pool, descriptor_sets)
}

unsafe fn get_semaphores(
    device: &Device,
    frames_in_flight: u32,
) -> Vec<(vk::Semaphore, vk::Semaphore, vk::Fence)> {
    (0..frames_in_flight)
        .map(|_| {
            let image_available = device
                .create_semaphore(&vk::SemaphoreCreateInfo::default(), None)
                .unwrap();
            let render_finished = device
                .create_semaphore(&vk::SemaphoreCreateInfo::default(), None)
                .unwrap();

            let fence_create_info =
                vk::FenceCreateInfo::builder().flags(vk::FenceCreateFlags::SIGNALED);
            let in_flight = device.create_fence(&fence_create_info, None).unwrap();

            (image_available, render_finished, in_flight)
        })
        .collect()
}

unsafe extern "system" fn vulkan_debug_callback(
    message_severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    message_type: vk::DebugUtilsMessageTypeFlagsEXT,
    p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    _user_data: *mut std::os::raw::c_void,
) -> vk::Bool32 {
    let callback_data = *p_callback_data;
    let message_id_number = callback_data.message_id_number;

    let message_id_name = if callback_data.p_message_id_name.is_null() {
        Cow::from("")
    } else {
        CStr::from_ptr(callback_data.p_message_id_name).to_string_lossy()
    };

    let message = if callback_data.p_message.is_null() {
        Cow::from("")
    } else {
        CStr::from_ptr(callback_data.p_message).to_string_lossy()
    };

    println!(
        "{message_severity:?}:\n{message_type:?} [{message_id_name} ({message_id_number})] : {message}\n",
    );

    vk::FALSE
}
