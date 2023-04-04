mod device_features;
pub use device_features::*;

use std::borrow::Cow;
use std::ffi::CStr;

use crate::prelude::*;
use crate::sync::get_device_queues;
use crate::sync::{QueueHandles, QueueRequest, QueueType};
use anyhow::{anyhow, Result};
use itertools::Itertools;
use raw_window_handle::HasRawDisplayHandle;
use raw_window_handle::HasRawWindowHandle;
use winit::window::WindowBuilder;

pub struct LoaderCreateInfo {
    pub width: u32,
    pub height: u32,
    pub title: String,
    pub device_features: DeviceFeaturesRequest,
    pub queue_requests: Vec<QueueRequest>,
}

pub struct Loader {
    pub window: Window,
    pub context: Context,
    pub entry: Entry,
    pub instance: Instance,
    pub debug: DebugUtils,
    pub surface: Surface,
    pub device: Device,
    pub allocator: Allocator,
    pub swapchain: Swapchain,
}

pub struct LoaderHandles {
    pub debug_messenger: vk::DebugUtilsMessengerEXT,
    pub surface: vk::SurfaceKHR,
    pub pdevice: vk::PhysicalDevice,
    pub queues: Vec<QueueHandles>,
}

impl Loader {
    pub fn new(loader_ci: LoaderCreateInfo) -> Result<(Self, LoaderHandles)> {
        unsafe {
            let (window, context) =
                get_window(loader_ci.width, loader_ci.height, &loader_ci.title)?;
            let (entry, instance) = get_instance(&window, &loader_ci.title)?;
            let (debug, debug_handle) = get_debug_hooks(&entry, &instance)?;
            let (surface, surface_handle) = get_surface(&window, &entry, &instance)?;
            let (pdevice_handle, device, queue_handles) = get_device(
                &instance,
                &surface,
                surface_handle,
                loader_ci.queue_requests,
                loader_ci.device_features,
            )?;
            let allocator = get_allocator(&instance, &device, pdevice_handle)?;
            let swapchain = Swapchain::new(&instance, &device);

            Ok((
                Self {
                    window,
                    context,
                    entry,
                    instance,
                    debug,
                    surface,
                    device,
                    allocator,
                    swapchain,
                },
                LoaderHandles {
                    debug_messenger: debug_handle,
                    surface: surface_handle,
                    pdevice: pdevice_handle,
                    queues: queue_handles,
                },
            ))
        }
    }
}

unsafe fn get_window(width: u32, height: u32, title: &str) -> Result<(Window, Context)> {
    let context = Context::new();

    let window = WindowBuilder::new()
        .with_title(title)
        .with_inner_size(winit::dpi::LogicalSize::new(width, height))
        .build(context.as_ref().as_ref().unwrap())?; // damn that's ugly

    Ok((window, context))
}

unsafe fn get_instance(window: &Window, title: &str) -> Result<(Entry, Instance)> {
    let entry = Entry::linked();

    let layer_names =
        vec![CStr::from_bytes_with_nul_unchecked(b"VK_LAYER_KHRONOS_validation\0").as_ptr()];

    #[allow(unused_mut)]
    let mut portability_extensions: Vec<*const i8> = vec![];
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    {
        portability_extensions.push(vk::KhrPortabilityEnumerationFn::name().as_ptr());
        portability_extensions.push(vk::KhrGetPhysicalDeviceProperties2Fn::name().as_ptr());
    }

    let extension_names = ash_window::enumerate_required_extensions(window.raw_display_handle())
        .unwrap()
        .iter()
        .chain(std::iter::once(&DebugUtils::name().as_ptr()))
        .chain(&portability_extensions)
        .map(|ptr| *ptr)
        .collect::<Vec<_>>();

    // Okay this sucks but we need the fucking null terminator
    let title_vec = title
        .as_bytes()
        .into_iter()
        .chain(b"\0")
        .map(|&b| b)
        .collect_vec();

    let app_info = vk::ApplicationInfo::builder()
        .application_name(CStr::from_bytes_with_nul_unchecked(&title_vec[..]))
        .application_version(vk::make_api_version(0, 1, 0, 0))
        .engine_name(CStr::from_bytes_with_nul_unchecked(b"silt\0"))
        .engine_version(vk::make_api_version(0, 0, 1, 0))
        .api_version(vk::make_api_version(0, 1, 0, 0));

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

    let instance = entry.create_instance(&instance_create_info, None)?;

    Ok((entry, instance))
}

unsafe fn get_debug_hooks(
    entry: &Entry,
    instance: &Instance,
) -> Result<(DebugUtils, vk::DebugUtilsMessengerEXT)> {
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
    let debug_callback = debug_utils_loader.create_debug_utils_messenger(&debug_info, None)?;

    Ok((debug_utils_loader, debug_callback))
}

unsafe fn get_surface(
    window: &Window,
    entry: &Entry,
    instance: &Instance,
) -> Result<(Surface, vk::SurfaceKHR)> {
    let surface_loader = Surface::new(entry, instance);

    let surface = ash_window::create_surface(
        entry,
        instance,
        window.raw_display_handle(),
        window.raw_window_handle(),
        None,
    )?;

    Ok((surface_loader, surface))
}

pub struct PhysicalDeviceInfo {
    pub pdevice: vk::PhysicalDevice,
    pub queues: Vec<vk::QueueFamilyProperties>,
    pub properties: vk::PhysicalDeviceProperties,
    pub features: vk::PhysicalDeviceFeatures,
}

unsafe fn get_device(
    instance: &Instance,
    surface_loader: &Surface,
    surface: vk::SurfaceKHR,
    queue_requests: Vec<QueueRequest>,
    device_features: DeviceFeaturesRequest,
) -> Result<(vk::PhysicalDevice, Device, Vec<QueueHandles>)> {
    if queue_requests.is_empty() {
        return Err(anyhow!(
            "no queues requested. you,,, you need queues to do things bestie"
        ));
    }

    let (info, queues, enabled_features) = instance
        .enumerate_physical_devices()?
        .into_iter()
        .map(|pdevice| {
            let queues = instance.get_physical_device_queue_family_properties(pdevice);
            let properties = instance.get_physical_device_properties(pdevice);
            let features = instance.get_physical_device_features(pdevice);

            PhysicalDeviceInfo {
                pdevice,
                queues,
                properties,
                features,
            }
        })
        .sorted_by_cached_key(|info| match info.properties.device_type {
            vk::PhysicalDeviceType::DISCRETE_GPU => 0,
            vk::PhysicalDeviceType::VIRTUAL_GPU => 1,
            vk::PhysicalDeviceType::INTEGRATED_GPU => 2,
            vk::PhysicalDeviceType::CPU => 3,
            _ => 4,
        })
        .filter_map(|info| {
            let (queues, failures) = queue_requests
                .iter()
                .map(|request| request.suitability(&info))
                .partition_result::<Vec<_>, Vec<_>, _, _>();

            if !failures.is_empty() {
                return None;
            }

            Some((info, queues))
        })
        .filter_map(|(info, queues)| {
            let supported_features: DeviceFeatures = info.features.into();

            if !supported_features.contains(device_features.required) {
                return None;
            }

            let enabled_features =
                (supported_features & (device_features.prefered | device_features.required)).into();

            let graphics_family = queues
                .iter()
                .find(|queue| queue.ty == QueueType::Graphics)?
                .family;

            if !surface_loader
                .get_physical_device_surface_support(info.pdevice, graphics_family, surface)
                .unwrap_or(false)
            {
                return None;
            }

            Some((info, queues, enabled_features))
        })
        .next()
        .ok_or(anyhow!("could not find suitable device"))?;

    let device_extensions_raw = [
        Swapchain::name().as_ptr(),
        #[cfg(any(target_os = "macos", target_os = "ios"))]
        vk::KhrPortabilitySubsetFn::name().as_ptr(),
    ];

    let device_ci = vk::DeviceCreateInfo::builder()
        .enabled_extension_names(&device_extensions_raw)
        .enabled_features(&enabled_features);

    let (device, queue_handles) = get_device_queues(instance, queues, info.pdevice, device_ci)?;

    Ok((info.pdevice, device, queue_handles))
}

unsafe fn get_allocator(
    instance: &Instance,
    device: &Device,
    pdevice: vk::PhysicalDevice,
) -> Result<Allocator> {
    let allocator_create_info = vk::AllocatorCreateInfo {
        physical_device: pdevice,
        device: device.clone(),
        instance: instance.clone(),
        debug_settings: Default::default(),
        buffer_device_address: false,
    };

    Allocator::new(&allocator_create_info)
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
