use crate::prelude::*;
use crate::loader::Loader;

pub fn get_msaa_samples(loader: &Loader, pdevice: vk::PhysicalDevice) -> vk::SampleCountFlags {
    let props = unsafe { loader.instance.get_physical_device_properties(pdevice) };

    props.limits.framebuffer_color_sample_counts.min(props.limits.framebuffer_depth_sample_counts)
}