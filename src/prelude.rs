pub use ash::extensions::{
    ext::DebugUtils,
    khr::{Surface, Swapchain},
};
pub use ash::{
    util::{Align, AlignIter},
    Device, Entry, Instance,
};
use shaderc::ShaderKind;
pub use winit::window::Window;

use anyhow::{anyhow, Result};
use std::{
    cell::{RefCell, Ref},
    collections::HashMap,
    ffi::c_void,
    num::NonZeroU64,
    ptr::NonNull,
    convert::From
};
use uuid::Uuid;
use winit::{
    event::Event,
    event_loop::{ControlFlow, EventLoop, EventLoopWindowTarget},
};

pub use crate::vk;
pub use crate::loader::Loader;

/// Subset of gpu_allocator::vulkan::Allocator with managed allocation handles
pub struct Allocator {
    inner: RefCell<gpu_allocator::vulkan::Allocator>,
    allocations: RefCell<HashMap<vk::Allocation, gpu_allocator::vulkan::Allocation>>,
}

impl Allocator {
    fn with<T>(
        &self,
        allocation: vk::Allocation,
        f: impl FnOnce(&gpu_allocator::vulkan::Allocation) -> T,
    ) -> Result<T> {
        Ok(f(self.allocations.borrow().get(&allocation).ok_or(
            anyhow!(
                "Allocation {} not found, possible use after free",
                Uuid::from(allocation).as_urn()
            ),
        )?))
    }

    fn with_result<T>(
        &self,
        allocation: vk::Allocation,
        f: impl FnOnce(&gpu_allocator::vulkan::Allocation) -> Result<T>,
    ) -> Result<T> {
        f(self.allocations.borrow().get(&allocation).ok_or(anyhow!(
            "Allocation {} not found, possible use after free",
            Uuid::from(allocation).as_urn()
        ))?)
    }

    pub fn new(desc: &vk::AllocatorCreateInfo) -> Result<Self> {
        Ok(Self {
            inner: RefCell::new(gpu_allocator::vulkan::Allocator::new(desc)?),
            allocations: Default::default(),
        })
    }

    pub fn allocate(&self, desc: &vk::AllocationCreateInfo<'_>) -> Result<vk::Allocation> {
        let allocation = self.inner.borrow_mut().allocate(desc)?;
        let uuid = Uuid::new_v4();
        if self
            .allocations
            .borrow_mut()
            .insert(uuid.into(), allocation)
            .is_some()
        {
            return Err(anyhow!("Allocation hanlde {} not unique", uuid.as_urn()));
        }
        Ok(uuid.into())
    }

    pub fn free(&self, allocation: vk::Allocation) -> Result<()> {
        let allocation = self
            .allocations
            .borrow_mut()
            .remove(&allocation)
            .ok_or(anyhow!(
                "Could not find allocation {}, possible double free error",
                Uuid::from(allocation).as_urn()
            ))?;
        self.inner.borrow_mut().free(allocation)?;
        Ok(())
    }

    pub fn get_chunk_id(&self, allocation: vk::Allocation) -> Result<NonZeroU64> {
        self.with_result(allocation, |a| {
            a.chunk_id().ok_or(anyhow!("could not obtain chunk id"))
        })
    }

    pub fn get_memory(&self, allocation: vk::Allocation) -> Result<vk::DeviceMemory> {
        self.with(allocation, |a| unsafe { a.memory() })
    }

    pub fn get_is_dedicated(&self, allocation: vk::Allocation) -> Result<bool> {
        self.with(allocation, gpu_allocator::vulkan::Allocation::is_dedicated)
    }

    pub fn get_offset(&self, allocation: vk::Allocation) -> Result<u64> {
        self.with(allocation, gpu_allocator::vulkan::Allocation::offset)
    }

    pub fn get_size(&self, allocation: vk::Allocation) -> Result<u64> {
        self.with(allocation, gpu_allocator::vulkan::Allocation::size)
    }

    pub fn get_mapped_ptr(&self, allocation: vk::Allocation) -> Result<NonNull<c_void>> {
        self.with_result(allocation, |a| {
            a.mapped_ptr().ok_or(anyhow!("memory not host visible"))
        })
    }

    pub fn get_is_null(&self, allocation: vk::Allocation) -> Result<bool> {
        self.with(allocation, gpu_allocator::vulkan::Allocation::is_null)
    }
}

pub struct Context {
    inner: RefCell<Option<EventLoop<()>>>,
}

impl Context {
    fn runner<S>(event: Event<'_, ()>, target: &EventLoopWindowTarget<()>, control_flow: &mut ControlFlow, mut draw: impl FnMut(&mut S), state: &mut S) {
        *control_flow = ControlFlow::Poll;
        match event {
            Event::RedrawEventsCleared => draw(state),
            _ => {}
        }
    }

    pub fn new() -> Self {
        Self {
            inner: RefCell::new(Some(EventLoop::new())),
        }
    }

    pub fn take(&self) -> Self {
        let inner = self.inner.take().unwrap();
        Self {
            inner: RefCell::new(Some(inner)),
        }
    }

    pub fn run<F>(self, event_handler: F)
    where
        F: 'static + FnMut(Event<'_, ()>, &EventLoopWindowTarget<()>, &mut ControlFlow),
    {
        self.inner.into_inner().unwrap().run(event_handler);
    }
    
    pub fn as_ref(&self) -> Ref<'_, Option<EventLoop<()>>> {
        self.inner.borrow()
    }
}



pub fn shader_kind_to_shader_stage_flags(kind: ShaderKind) -> vk::ShaderStageFlags {
    match kind {
        ShaderKind::Vertex => vk::ShaderStageFlags::VERTEX,
        ShaderKind::Fragment => vk::ShaderStageFlags::FRAGMENT,
        ShaderKind::Compute => vk::ShaderStageFlags::COMPUTE,
        ShaderKind::Geometry => vk::ShaderStageFlags::GEOMETRY,
        ShaderKind::TessControl => vk::ShaderStageFlags::TESSELLATION_CONTROL,
        ShaderKind::TessEvaluation => vk::ShaderStageFlags::TESSELLATION_EVALUATION,
        ShaderKind::DefaultVertex => vk::ShaderStageFlags::VERTEX,
        ShaderKind::DefaultFragment => vk::ShaderStageFlags::FRAGMENT,
        ShaderKind::DefaultCompute => vk::ShaderStageFlags::COMPUTE,
        ShaderKind::DefaultGeometry => vk::ShaderStageFlags::GEOMETRY,
        ShaderKind::DefaultTessControl => vk::ShaderStageFlags::TESSELLATION_CONTROL,
        ShaderKind::DefaultTessEvaluation => vk::ShaderStageFlags::TESSELLATION_EVALUATION,
        ShaderKind::RayGeneration => vk::ShaderStageFlags::RAYGEN_KHR,
        ShaderKind::AnyHit => vk::ShaderStageFlags::ANY_HIT_KHR,
        ShaderKind::ClosestHit => vk::ShaderStageFlags::CLOSEST_HIT_KHR,
        ShaderKind::Miss => vk::ShaderStageFlags::MISS_KHR,
        ShaderKind::Intersection => vk::ShaderStageFlags::INTERSECTION_KHR,
        ShaderKind::Callable => vk::ShaderStageFlags::CALLABLE_KHR,
        ShaderKind::DefaultRayGeneration => vk::ShaderStageFlags::RAYGEN_KHR,
        ShaderKind::DefaultAnyHit => vk::ShaderStageFlags::ANY_HIT_KHR,
        ShaderKind::DefaultClosestHit => vk::ShaderStageFlags::CLOSEST_HIT_KHR,
        ShaderKind::DefaultMiss => vk::ShaderStageFlags::MISS_KHR,
        ShaderKind::DefaultIntersection => vk::ShaderStageFlags::INTERSECTION_KHR,
        ShaderKind::DefaultCallable => vk::ShaderStageFlags::CALLABLE_KHR,
        ShaderKind::Task => vk::ShaderStageFlags::TASK_EXT,
        ShaderKind::Mesh => vk::ShaderStageFlags::MESH_EXT,
        ShaderKind::DefaultTask => vk::ShaderStageFlags::TASK_EXT,
        ShaderKind::DefaultMesh => vk::ShaderStageFlags::MESH_EXT,
        _ => panic!("Unknown shader kind")
    }
}

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