use std::{cell::RefCell, collections::HashMap};

use crate::{prelude::*, storage::descriptors::{Layouts, DescriptorFrequency}, resources::{Parity, ParitySet}};

pub struct RenderContext {
    pub loader: Loader,
    pub parity: Parity,
    pub frame: usize,
    pub command_buffers: ParitySet<vk::CommandBuffer>,
    pub swapchain: RefCell<Swapchain>,
    pub present_pass: vk::RenderPass,
    pub pipeline: vk::Pipeline,
    pub layouts: Layouts,
    pub descriptors: HashMap<DescriptorFrequency, ParitySet<vk::DescriptorSet>>
}