use ash::extensions::ext::DebugUtils;
use ash::extensions::khr::{Surface, Swapchain};
use ash::util::{read_spv, Align};
use ash::vk::{self, PhysicalDeviceType};
use ash::{Device, Entry, Instance};
use gpu_allocator::vulkan as vma;
use itertools::Itertools;
use memoffset::offset_of;
use nalgebra_glm as glm;
use once_cell::sync::Lazy;
use raw_window_handle::{HasRawDisplayHandle, HasRawWindowHandle};
use std::borrow::Cow;
use std::cell::RefCell;
use std::ffi::CStr;
use std::time::Instant;
use std::{fs, path::Path};
use winit::event::{Event, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::window::{Window, WindowBuilder};

static START_TIME: Lazy<Instant> = Lazy::new(|| Instant::now());

pub type QueueFamilyIndex = u32;

#[derive(Copy, Clone, Debug, Default)]
pub struct Vertex {
    pub pos: glm::Vec2,
    pub color: glm::Vec3,
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
                .format(vk::Format::R32G32_SFLOAT)
                .offset(offset_of!(Vertex, pos) as u32)
                .build(),
            vk::VertexInputAttributeDescription::builder()
                .binding(0)
                .location(1)
                .format(vk::Format::R32G32B32_SFLOAT)
                .offset(offset_of!(Vertex, color) as u32)
                .build(),
        ]
    }
}

#[derive(Clone, Copy)]
#[allow(dead_code)]
pub struct UniformBufferObject {
    model: glm::Mat4,
    view: glm::Mat4,
    projection: glm::Mat4,
}

pub struct FrameData {
    pub command_buffer: vk::CommandBuffer,
    pub uniform_buffer: vk::Buffer,
    pub uniform_buffer_allocation: vma::Allocation,
    pub uniform_buffer_mapping: RefCell<Align<UniformBufferObject>>,
    pub descriptor_set: vk::DescriptorSet,
    pub image_available: vk::Semaphore,
    pub render_finished: vk::Semaphore,
    pub in_flight: vk::Fence,
}

pub struct VulkanData {
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u16>,

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
    pub queue_family_index: u32,
    pub queue: vk::Queue,

    pub allocator: RefCell<vma::Allocator>,

    pub surface_capabilities: vk::SurfaceCapabilitiesKHR,
    pub surface_format: vk::SurfaceFormatKHR,
    pub present_mode: vk::PresentModeKHR,

    pub swapchain_loader: Option<Swapchain>,
    pub swapchain: Option<vk::SwapchainKHR>,
    pub image_extent: vk::Extent2D,

    pub images: Vec<vk::Image>,
    pub image_views: Vec<vk::ImageView>,

    pub render_pass: vk::RenderPass,
    pub shader_modules: Vec<vk::ShaderModule>,
    pub pipeline_layout: vk::PipelineLayout,
    pub pipeline: vk::Pipeline,

    pub framebuffers: Vec<vk::Framebuffer>,

    pub vertex_buffer: vk::Buffer,
    pub vertex_buffer_allocation: vma::Allocation,
    pub index_buffer: vk::Buffer,
    pub index_buffer_allocation: vma::Allocation,

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

            let vertices = vec![
                Vertex {
                    pos: glm::vec2(-0.5, -0.5),
                    color: glm::vec3(1., 0., 0.),
                },
                Vertex {
                    pos: glm::vec2(0.5, -0.5),
                    color: glm::vec3(0., 1., 0.),
                },
                Vertex {
                    pos: glm::vec2(0.5, 0.5),
                    color: glm::vec3(0., 0., 1.),
                },
                Vertex {
                    pos: glm::vec2(-0.5, 0.5),
                    color: glm::vec3(1., 1., 1.),
                },
            ];

            let indices = vec![0, 1, 2, 2, 3, 0];

            let (window, event_loop) = get_window(width, height);
            let (entry, instance) = get_instance(&window);
            let (debug_utils_loader, debug_callback) = get_debug_hooks(&entry, &instance);
            let (surface_loader, surface) = get_surface(&window, &entry, &instance);
            let (pdevice, device, queue_family_index, queue) =
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
            let render_pass = get_uniform_buffers_render_pass(&device, surface_format);
            let (shader_modules, descriptor_set_layout, pipeline_layout, pipeline) =
                get_uniform_buffers_pipeline(&device, &render_pass);
            let framebuffers = get_framebuffers(&device, &image_views, image_extent, &render_pass);
            let (command_pool, transient_command_pool, command_buffers) =
                get_command_buffers(&device, queue_family_index, frames_in_flight);
            let (vertex_buffer, vertex_buffer_allocation) = get_vertex_buffer(
                &device,
                &queue,
                &transient_command_pool,
                &mut allocator.borrow_mut(),
                &vertices,
            );
            let (index_buffer, index_buffer_allocation) = get_index_buffer(
                &device,
                &queue,
                &transient_command_pool,
                &mut allocator.borrow_mut(),
                &indices,
            );
            let uniform_buffers =
                get_uniform_buffers(&device, &mut allocator.borrow_mut(), frames_in_flight);
            let (descriptor_pool, descriptor_sets) =
                get_descriptor_sets(&device, &descriptor_set_layout, &uniform_buffers);
            let sync_objects = get_semaphores(&device, frames_in_flight);

            let frame_data = itertools::multizip((command_buffers, uniform_buffers, descriptor_sets, sync_objects))
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
                swapchain_loader: Some(swapchain_loader),
                swapchain: Some(swapchain),
                image_extent,
                images,
                image_views,
                render_pass,
                shader_modules,
                pipeline_layout,
                pipeline,
                framebuffers,
                vertex_buffer,
                vertex_buffer_allocation,
                index_buffer,
                index_buffer_allocation,
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

        let clear_color = vk::ClearValue {
            color: vk::ClearColorValue {
                float32: [0., 0., 0., 0.],
            },
        };

        let render_pass_info = vk::RenderPassBeginInfo::builder()
            .render_pass(self.render_pass)
            .framebuffer(*self.framebuffers.get(image_index as usize).unwrap())
            .render_area(vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: self.image_extent,
            })
            .clear_values(std::slice::from_ref(&clear_color));
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
            vk::IndexType::UINT16,
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

        self.device.cmd_bind_descriptor_sets(frame.command_buffer, vk::PipelineBindPoint::GRAPHICS, self.pipeline_layout, 0, &[frame.descriptor_set], &[]);

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

    unsafe fn draw_frame(&mut self, current_frame: usize) {
        let frame = &self.frame_data[current_frame];

        self.device
            .wait_for_fences(&[frame.in_flight], true, u64::MAX)
            .unwrap();
        self.device.reset_fences(&[frame.in_flight]).unwrap();

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

        self.device
            .reset_command_buffer(frame.command_buffer, vk::CommandBufferResetFlags::empty())
            .unwrap();
        self.record_command_buffer(image_index, current_frame);

        let time = START_TIME.elapsed().as_secs_f32();
        let ubo = UniformBufferObject {
            model: glm::rotation(time, &glm::vec3(0., 0., 1.)),
            view: glm::look_at(
                &glm::vec3(2., 2., 2.),
                &glm::vec3(0., 0., 0.),
                &glm::vec3(0., 0., 1.),
            ),
            projection: {
                let mut gl_formatted = glm::perspective(self.aspect_ratio(), 90., 0.1, 10.);
                *gl_formatted.get_mut((1, 1)).unwrap() *= -1.;
                gl_formatted
            },
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
        event_loop.run(move |event, _, control_flow| {
            *control_flow = ControlFlow::Poll;
            match event {
                Event::WindowEvent {
                    event: WindowEvent::CloseRequested,
                    ..
                } => *control_flow = ControlFlow::Exit,
                Event::MainEventsCleared => unsafe {
                    self.current_frame = (self.current_frame + 1) % self.frame_data.len();
                    let new_time = Instant::now();
                    let diff = new_time - old_time;
                    old_time = new_time;
                    println!("Frame Time: {:.2}ms", diff.as_micros() as f32 / 1000.);
                    self.draw_frame(self.current_frame);
                },
                _ => (),
            }
        });
    }

    pub unsafe fn recreate_swapchain(&mut self) {
        self.device.device_wait_idle().unwrap();

        let size = self.window.inner_size();
        self.framebuffers.clear();
        self.image_views.clear();
        self.images.clear();
        self.swapchain_loader
            .as_ref()
            .unwrap()
            .destroy_swapchain(self.swapchain.unwrap(), None);
        self.swapchain = None;
        self.swapchain_loader = None;

        loop {
            self.surface_capabilities =
                get_surface_properties(&self.surface_loader, &self.surface, &self.pdevice).0;
            if self.surface_capabilities.current_extent.width != 0 {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(500));
        }
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
        let framebuffers =
            get_framebuffers(&self.device, &image_views, image_extent, &self.render_pass);

        self.swapchain_loader = Some(swapchain_loader);
        self.swapchain = Some(swapchain);
        self.image_extent = image_extent;
        self.images = images;
        self.image_views = image_views;
        self.framebuffers = framebuffers;
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

unsafe fn get_window(width: u32, height: u32) -> (Window, EventLoop<()>) {
    let event_loop = EventLoop::new();

    let window = WindowBuilder::new()
        .with_title("Silt Triangle")
        .with_inner_size(winit::dpi::LogicalSize::new(width, height))
        .build(&event_loop)
        .unwrap();

    (window, event_loop)
}

unsafe fn get_instance(window: &Window) -> (Entry, Instance) {
    let entry = Entry::linked();

    let layer_names =
        vec![CStr::from_bytes_with_nul_unchecked(b"VK_LAYER_KHRONOS_validation\0").as_ptr()];
    let extension_names = ash_window::enumerate_required_extensions(window.raw_display_handle())
        .unwrap()
        .iter()
        .chain(std::iter::once(&DebugUtils::name().as_ptr()))
        .map(|ptr| *ptr)
        .collect::<Vec<_>>();

    let app_info = vk::ApplicationInfo::builder()
        .application_name(CStr::from_bytes_with_nul_unchecked(b"Silt Triangle\0"))
        .application_version(vk::make_api_version(0, 1, 0, 0))
        .engine_name(CStr::from_bytes_with_nul_unchecked(b"silt\0"))
        .engine_version(vk::make_api_version(0, 0, 1, 0))
        .api_version(vk::make_api_version(0, 1, 2, 0));

    let instance_create_info = vk::InstanceCreateInfo::builder()
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

unsafe fn get_device_and_queue(
    instance: &Instance,
    surface_loader: &Surface,
    surface: &vk::SurfaceKHR,
) -> (vk::PhysicalDevice, Device, QueueFamilyIndex, vk::Queue) {
    let (pdevice, queue_family_index, _) = instance
        .enumerate_physical_devices()
        .unwrap()
        .iter()
        .filter_map(|pdevice| {
            instance
                .get_physical_device_queue_family_properties(*pdevice)
                .iter()
                .enumerate()
                .find_map(|(index, info)| {
                    let supports_graphic_and_surface = info
                        .queue_flags
                        .contains(vk::QueueFlags::GRAPHICS)
                        && surface_loader
                            .get_physical_device_surface_support(*pdevice, index as u32, *surface)
                            .unwrap();
                    if supports_graphic_and_surface {
                        Some((*pdevice, index as u32))
                    } else {
                        None
                    }
                })
        })
        .map(|(pdevice, index)| {
            let priority = match instance.get_physical_device_properties(pdevice).device_type {
                PhysicalDeviceType::DISCRETE_GPU => 0,
                PhysicalDeviceType::INTEGRATED_GPU => 1,
                _ => i32::MAX,
            };
            (pdevice, index, priority)
        })
        .sorted_by_key(|(_, _, priority)| *priority)
        .next()
        .expect("Couldn't find suitable device.");

    let device_extensions_raw = [Swapchain::name().as_ptr()];

    let queue_priorities = [1.0];
    let queue_create_info = vk::DeviceQueueCreateInfo::builder()
        .queue_family_index(queue_family_index)
        .queue_priorities(&queue_priorities);

    let device_create_info = vk::DeviceCreateInfo::builder()
        .enabled_extension_names(&device_extensions_raw)
        .queue_create_infos(std::slice::from_ref(&queue_create_info));

    let device = instance
        .create_device(pdevice, &device_create_info, None)
        .unwrap();
    let queue = device.get_device_queue(queue_family_index, 0);

    (pdevice, device, queue_family_index, queue)
}

unsafe fn get_allocator(
    instance: &Instance,
    device: &Device,
    pdevice: &vk::PhysicalDevice,
) -> RefCell<vma::Allocator> {
    let allocator_create_info = vma::AllocatorCreateDesc {
        physical_device: *pdevice,
        device: device.clone(),
        instance: instance.clone(),
        debug_settings: Default::default(),
        buffer_device_address: true,
    };

    RefCell::new(vma::Allocator::new(&allocator_create_info).unwrap())
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
    let surface_format = *surface_loader
        .get_physical_device_surface_formats(*pdevice, *surface)
        .unwrap()
        .get(0)
        .unwrap();
    let present_mode = surface_loader
        .get_physical_device_surface_present_modes(*pdevice, *surface)
        .unwrap()
        .into_iter()
        .find(|&mode| mode == vk::PresentModeKHR::MAILBOX)
        .unwrap_or(vk::PresentModeKHR::FIFO);

    (surface_capabilities, surface_format, present_mode)
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
    let code = get_shader_code(name);
    let shader_module_create_info = vk::ShaderModuleCreateInfo::builder().code(&code);

    device
        .create_shader_module(&shader_module_create_info, None)
        .unwrap()
}

fn get_shader_code(name: impl AsRef<str>) -> Vec<u32> {
    let path = Path::new(env!("OUT_DIR")).join(format!("{}.spirv", name.as_ref()));
    let mut file = fs::File::open(path).unwrap();
    read_spv(&mut file).unwrap()
}

unsafe fn get_uniform_buffers_render_pass(
    device: &Device,
    surface_format: vk::SurfaceFormatKHR,
) -> vk::RenderPass {
    let color_attachment = vk::AttachmentDescription::builder()
        .format(surface_format.format)
        .samples(vk::SampleCountFlags::TYPE_1)
        .load_op(vk::AttachmentLoadOp::CLEAR)
        .store_op(vk::AttachmentStoreOp::STORE)
        .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
        .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .final_layout(vk::ImageLayout::PRESENT_SRC_KHR);

    let color_attachment_reference = vk::AttachmentReference {
        attachment: 0,
        layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
    };

    let subpass = vk::SubpassDescription::builder()
        .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
        .color_attachments(std::slice::from_ref(&color_attachment_reference));

    let subpass_dependency = vk::SubpassDependency::builder()
        .src_subpass(vk::SUBPASS_EXTERNAL)
        .dst_subpass(0)
        .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
        .src_access_mask(vk::AccessFlags::empty())
        .dst_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
        .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE);

    let render_pass_create_info = vk::RenderPassCreateInfo::builder()
        .attachments(std::slice::from_ref(&color_attachment))
        .subpasses(std::slice::from_ref(&subpass))
        .dependencies(std::slice::from_ref(&subpass_dependency));

    let render_pass = device
        .create_render_pass(&render_pass_create_info, None)
        .unwrap();

    render_pass
}

unsafe fn get_uniform_buffers_pipeline(
    device: &Device,
    render_pass: &vk::RenderPass,
) -> (
    Vec<vk::ShaderModule>,
    vk::DescriptorSetLayout,
    vk::PipelineLayout,
    vk::Pipeline,
) {
    let vertex_shader_module = get_shader_module(device, "uniform_buffers.vert");
    let fragment_shader_module = get_shader_module(device, "vertex_buffers.frag");

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
        .cull_mode(vk::CullModeFlags::BACK)
        .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
        .depth_bias_enable(false);

    let multisample_state = vk::PipelineMultisampleStateCreateInfo::builder()
        .sample_shading_enable(false)
        .rasterization_samples(vk::SampleCountFlags::TYPE_1);

    let color_blend_attachment_state = vk::PipelineColorBlendAttachmentState::builder()
        .color_write_mask(vk::ColorComponentFlags::RGBA)
        .blend_enable(false);

    let color_blend_state = vk::PipelineColorBlendStateCreateInfo::builder()
        .logic_op_enable(false)
        .attachments(std::slice::from_ref(&color_blend_attachment_state));

    let descriptor_set_layout_binding = vk::DescriptorSetLayoutBinding::builder()
        .binding(0)
        .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
        .descriptor_count(1)
        .stage_flags(vk::ShaderStageFlags::VERTEX);

    let descriptor_set_layout_create_info = vk::DescriptorSetLayoutCreateInfo::builder()
        .bindings(std::slice::from_ref(&descriptor_set_layout_binding));

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
    render_pass: &vk::RenderPass,
) -> Vec<vk::Framebuffer> {
    image_views
        .iter()
        .map(|image_view| {
            let framebuffer_create_info = vk::FramebufferCreateInfo::builder()
                .render_pass(*render_pass)
                .attachments(std::slice::from_ref(image_view))
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
    allocator: &mut vma::Allocator,
    size: u64,
    usage: vk::BufferUsageFlags,
    location: gpu_allocator::MemoryLocation,
) -> (vk::Buffer, vma::Allocation, vk::MemoryRequirements) {
    let buffer_create_info = vk::BufferCreateInfo::builder()
        .size(size)
        .usage(usage)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);

    let buffer = device.create_buffer(&buffer_create_info, None).unwrap();
    let requirements = device.get_buffer_memory_requirements(buffer);

    let allocation_create_info = vma::AllocationCreateDesc {
        name: "UNNAMED BUFFER",
        requirements,
        location,
        linear: true,
        allocation_scheme: vma::AllocationScheme::DedicatedBuffer(buffer),
    };

    let allocation = allocator.allocate(&allocation_create_info).unwrap();
    device
        .bind_buffer_memory(buffer, allocation.memory(), allocation.offset())
        .unwrap();
    (buffer, allocation, requirements)
}

unsafe fn copy_buffer(
    device: &Device,
    queue: &vk::Queue,
    command_pool: &vk::CommandPool,
    src: vk::Buffer,
    dst: vk::Buffer,
    size: u64,
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

    let region = vk::BufferCopy::builder().size(size).build();
    device.cmd_copy_buffer(command_buffer, src, dst, &[region]);
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

unsafe fn map_to_buffer<T: Copy>(
    device: &Device,
    allocation: &vma::Allocation,
    requirements: vk::MemoryRequirements,
    slice: &[T],
) {
    let ptr = device
        .map_memory(
            allocation.memory(),
            0,
            requirements.size,
            vk::MemoryMapFlags::empty(),
        )
        .unwrap();

    let mut align = Align::new(ptr, std::mem::align_of::<T>() as u64, requirements.size);
    align.copy_from_slice(slice);
    device.unmap_memory(allocation.memory());
}

unsafe fn persistent_map_to_buffer<T>(
    device: &Device,
    allocation: &vma::Allocation,
    requirements: vk::MemoryRequirements,
) -> Align<T> {
    let ptr = device
        .map_memory(
            allocation.memory(),
            0,
            requirements.size,
            vk::MemoryMapFlags::empty(),
        )
        .unwrap();
    Align::new(ptr, std::mem::align_of::<T>() as u64, requirements.size)
}

unsafe fn get_vertex_buffer(
    device: &Device,
    queue: &vk::Queue,
    command_pool: &vk::CommandPool,
    allocator: &mut vma::Allocator,
    vertices: &[Vertex],
) -> (vk::Buffer, vma::Allocation) {
    let (src_buffer, src_allocation, src_requirements) = create_buffer(
        device,
        allocator,
        std::mem::size_of_val(vertices) as u64,
        vk::BufferUsageFlags::TRANSFER_SRC,
        gpu_allocator::MemoryLocation::CpuToGpu,
    );

    let (buffer, allocation, requirements) = create_buffer(
        device,
        allocator,
        std::mem::size_of_val(vertices) as u64,
        vk::BufferUsageFlags::VERTEX_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
        gpu_allocator::MemoryLocation::GpuOnly,
    );

    map_to_buffer(device, &src_allocation, src_requirements, vertices);
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
    allocator: &mut vma::Allocator,
    indices: &[u16],
) -> (vk::Buffer, vma::Allocation) {
    let (src_buffer, src_allocation, src_requirements) = create_buffer(
        device,
        allocator,
        std::mem::size_of_val(indices) as u64,
        vk::BufferUsageFlags::TRANSFER_SRC,
        gpu_allocator::MemoryLocation::CpuToGpu,
    );

    let (buffer, allocation, requirements) = create_buffer(
        device,
        allocator,
        std::mem::size_of_val(indices) as u64,
        vk::BufferUsageFlags::INDEX_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
        gpu_allocator::MemoryLocation::GpuOnly,
    );

    map_to_buffer(device, &src_allocation, src_requirements, indices);
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

unsafe fn get_uniform_buffers(
    device: &Device,
    allocator: &mut vma::Allocator,
    frames_in_flight: u32,
) -> Vec<(vk::Buffer, vma::Allocation, Align<UniformBufferObject>)> {
    (0..frames_in_flight)
        .map(|_| {
            let (buffer, allocation, requirements) = create_buffer(
                device,
                allocator,
                std::mem::size_of::<UniformBufferObject>() as u64,
                vk::BufferUsageFlags::UNIFORM_BUFFER,
                gpu_allocator::MemoryLocation::CpuToGpu,
            );

            let persistent_mapping = persistent_map_to_buffer(device, &allocation, requirements);

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
        .queue_family_index(queue_family_index);

    let command_pool = device
        .create_command_pool(&command_pool_create_info, None)
        .unwrap();

    let transient_command_pool_create_info = vk::CommandPoolCreateInfo::builder()
        .flags(vk::CommandPoolCreateFlags::TRANSIENT)
        .queue_family_index(queue_family_index);

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
    uniform_buffers: &Vec<(vk::Buffer, vma::Allocation, Align<UniformBufferObject>)>,
) -> (vk::DescriptorPool, Vec<vk::DescriptorSet>) {
    let frames_in_flight = uniform_buffers.len() as u32;

    let pool_sizes = [vk::DescriptorPoolSize::builder()
        .descriptor_count(frames_in_flight)
        .build()];

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

        let descriptor_write = vk::WriteDescriptorSet::builder()
            .buffer_info(std::slice::from_ref(&buffer_info))
            .dst_set(*descriptor_set)
            .dst_binding(0)
            .dst_array_element(0)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER);

        device.update_descriptor_sets(std::slice::from_ref(&descriptor_write), &[]);
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
