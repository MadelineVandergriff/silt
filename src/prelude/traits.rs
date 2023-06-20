use crate::vk;

use super::Loader;

pub trait Destructible {
    fn destroy(self, loader: &Loader);
}

impl Destructible for vk::Buffer {
    fn destroy(self, loader: &Loader) {
        unsafe { loader.device.destroy_buffer(self, None) };
    }
}

impl Destructible for vk::BufferView {
    fn destroy(self, loader: &Loader) {
        unsafe { loader.device.destroy_buffer_view(self, None) };
    }
}

impl Destructible for vk::CommandPool  {
    fn destroy(self, loader: &Loader) {
        unsafe { loader.device.destroy_command_pool(self, None) };
    }
}

impl Destructible for vk::DescriptorPool {
    fn destroy(self, loader: &Loader) {
        unsafe { loader.device.destroy_descriptor_pool(self, None) };
    }
}

impl Destructible for vk::DescriptorSetLayout {
    fn destroy(self, loader: &Loader) {
        unsafe { loader.device.destroy_descriptor_set_layout(self, None) };
    }
}

impl Destructible for vk::Event {
    fn destroy(self, loader: &Loader) {
        unsafe { loader.device.destroy_event(self, None) };
    }
}

impl Destructible for vk::Fence {
    fn destroy(self, loader: &Loader) {
        unsafe { loader.device.destroy_fence(self, None) };
    }
}

impl Destructible for vk::Framebuffer {
    fn destroy(self, loader: &Loader) {
        unsafe { loader.device.destroy_framebuffer(self, None) };
    }
}

impl Destructible for vk::Image {
    fn destroy(self, loader: &Loader) {
        unsafe { loader.device.destroy_image(self, None) };
    }
}

impl Destructible for vk::ImageView {
    fn destroy(self, loader: &Loader) {
        unsafe { loader.device.destroy_image_view(self, None) };
    }
}

impl Destructible for vk::Pipeline {
    fn destroy(self, loader: &Loader) {
        unsafe { loader.device.destroy_pipeline(self, None) };
    }
}

impl Destructible for vk::PipelineCache {
    fn destroy(self, loader: &Loader) {
        unsafe { loader.device.destroy_pipeline_cache(self, None) };
    }
}

impl Destructible for vk::PipelineLayout {
    fn destroy(self, loader: &Loader) {
        unsafe { loader.device.destroy_pipeline_layout(self, None) };
    }
}

impl Destructible for vk::RenderPass {
    fn destroy(self, loader: &Loader) {
        unsafe { loader.device.destroy_render_pass(self, None) };
    }
}

impl Destructible for vk::Sampler {
    fn destroy(self, loader: &Loader) {
        unsafe { loader.device.destroy_sampler(self, None) };
    }
}

impl Destructible for vk::Semaphore {
    fn destroy(self, loader: &Loader) {
        unsafe { loader.device.destroy_semaphore(self, None) };
    }
}

impl Destructible for vk::ShaderModule {
    fn destroy(self, loader: &Loader) {
        unsafe { loader.device.destroy_shader_module(self, None) };
    }
}

impl Destructible for vk::Allocation {
    fn destroy(self, loader: &Loader) {
        loader.allocator.free(self).unwrap();
    }
}

impl Destructible for vk::SurfaceKHR {
    fn destroy(self, loader: &Loader) {
        unsafe { loader.surface.destroy_surface(self, None) };
    }
}

impl Destructible for vk::SwapchainKHR {
    fn destroy(self, loader: &Loader) {
        unsafe { loader.swapchain.destroy_swapchain(self, None) };
    }
}

pub trait IterDestructible<T: Destructible>: IntoIterator<Item = T> + Sized {
    fn destroy(self, loader: &Loader) {
        self.into_iter().for_each(|value| value.destroy(loader));
    }
}

pub trait Vectorizable<T> {
    fn to_vec(self) -> Vec<T>;
}

impl<T> Vectorizable<T> for T {
    fn to_vec(self) -> Vec<T> {
        vec![self]
    }
}

impl<T> Vectorizable<T> for Vec<T> {
    fn to_vec(self) -> Vec<T> {
        self
    }
}

impl<T: Clone> Vectorizable<T> for &[T] {
    fn to_vec(self) -> Vec<T> {
        Vec::from(self)
    }
}

macro_rules! impl_vectorizable_for_array {
    ($( $size:literal ),*) => {
        $(
        impl<T> Vectorizable<T> for [T; $size] {
            fn to_vec(self) -> Vec<T> {
                self.into()
            }
        }
        )*
    };
}

impl_vectorizable_for_array!(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);