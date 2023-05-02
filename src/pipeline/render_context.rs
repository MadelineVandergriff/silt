use std::{cell::RefCell, collections::HashMap};

use crate::{prelude::*, storage::descriptors::{Layouts, DescriptorFrequency}, model::Model};

pub struct RenderContext<'a> {
    pub loader: &'a Loader,
    pub parity: Parity,
    pub frame: usize,
    pub command_buffers: ParitySet<vk::CommandBuffer>,
    pub swapchain: &'a RefCell<Swapchain>,
    pub present_pass: vk::RenderPass,
    pub pipeline: vk::Pipeline,
    pub layouts: Layouts,
    pub descriptors: HashMap<DescriptorFrequency, ParitySet<vk::DescriptorSet>>
}