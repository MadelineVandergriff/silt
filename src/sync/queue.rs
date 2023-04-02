use crate::prelude::*;
use crate::vk;
use anyhow::{anyhow, Result};
use itertools::Itertools;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QueueType {
    Graphics,
    Compute,
    Transfer,
    SparseBinding,
}

pub struct QueueRequest {
    pub ty: QueueType,
    pub count: u32,
}

pub struct QueueProperties {
    pub ty: QueueType,
    pub count: u32,
    pub family: u32,
    pub props: vk::QueueFamilyProperties,
}

#[derive(Clone, Default)]
struct QueueAllocator {
    pub requested: u32,
    pub provided: u32,
    pub requests: u32,
    given: Vec<u32>,
}

impl QueueAllocator {
    pub fn new() -> Self {
        Default::default()
    }

    fn has_unique(&self) -> bool {
        self.requests <= self.provided
    }

    pub fn fufill(&mut self, request: u32) -> Vec<u32> {
        if !self.has_unique() && !self.given.contains(&0) {
            self.given.push(0);
        }

        let expected_fufillment = if self.has_unique() {
            1 + (request - 1) * (self.provided - self.requests) / self.requested
        } else {
            1 + (request - 1) * (self.provided - 1) / self.requested
        };

        let state = if self.has_unique() { vec![] } else { vec![0] };

        (0..self.provided).fold(state, |mut state, idx| {
            if (state.len() as u32) < expected_fufillment && !self.given.contains(&idx) {
                self.given.push(idx);
                state.push(idx);
            }

            state
        })
    }
}

/// All queues but the initial one are guaranteed to be unique
pub struct QueueHandles {
    pub ty: QueueType,
    pub queues: Vec<vk::Queue>,
    pub family: u32,
}

impl QueueType {
    pub fn suitability(&self) -> Result<QueueProperties> {
        match self {
            QueueType::Graphics => {}
            QueueType::Compute => {}
            QueueType::Transfer => {}
            QueueType::SparseBinding => {}
        };
        Err(anyhow!(""))
    }
}

pub fn get_queue_ci(
    properties: Vec<QueueProperties>,
    instance: &Instance,
    pdevice: vk::PhysicalDevice,
    builder: vk::DeviceCreateInfoBuilder<'_>,
) -> () {
    let max_queue_count = properties
        .iter()
        .map(|&prop| prop.props.queue_count)
        .max()
        .unwrap();
    let priorities = vec![1.0; max_queue_count as usize];

    let mut queue_allocators = properties
        .iter()
        .into_grouping_map_by(|prop| prop.family)
        .fold(QueueAllocator::new(), |mut alloc, _, prop| {
            alloc.requests += 1;
            alloc.requested += prop.count;
            alloc.provided = alloc.requested.min(prop.props.queue_count);
            alloc
        });

    let queue_ci = queue_allocators
        .iter()
        .map(|(&queue_family_index, alloc)| {
            vk::DeviceQueueCreateInfo::builder()
                .queue_family_index(queue_family_index)
                .queue_priorities(&priorities[..alloc.provided as usize])
                .build() // SAFETY: Priorities is not dropped until end of function
        })
        .collect_vec();

    let device_ci = builder.queue_create_infos(&queue_ci[..]);
    let device = unsafe { instance.create_device(pdevice, &device_ci, None).unwrap() };

    let queues = properties
        .into_iter()
        .map(|prop| {
            let queues = queue_allocators
                .get_mut(&prop.family)
                .unwrap()
                .fufill(prop.count)
                .into_iter()
                .map(|idx| unsafe { device.get_device_queue(prop.family, idx) })
                .collect_vec();

            QueueHandles {
                ty: prop.ty,
                queues,
                family: prop.family,
            }
        })
        .collect_vec();
}
