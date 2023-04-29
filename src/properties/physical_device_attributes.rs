use crate::prelude::*;
use crate::loader::Loader;

pub fn get_sample_counts(loader: &Loader, pdevice: vk::PhysicalDevice) -> vk::SampleCountFlags {
    let limits = unsafe { loader.instance.get_physical_device_properties(pdevice).limits };
    let samples = limits.framebuffer_color_sample_counts.min(limits.framebuffer_depth_sample_counts);
    vk::SampleCountFlags::from_raw((samples.as_raw() + 1) >> 1)
}