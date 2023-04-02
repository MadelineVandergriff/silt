use crate::vk;
use crate::prelude::*;
use anyhow::{Result, anyhow};
use itertools::Itertools;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QueueType {
    Graphics, Compute, Transfer, SparseBinding
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

/// All queues but the final one are guaranteed to be unique
pub struct QueueHandles {
    pub ty: QueueType,
    pub queues: Vec<vk::Queue>,
    pub family: u32,
}

impl QueueType {
    pub fn suitability(&self) -> Result<QueueProperties> {
        match self {
            QueueType::Graphics => {

            },
            QueueType::Compute => {

            },
            QueueType::Transfer => {

            },
            QueueType::SparseBinding => {

            },
        }; 5
    }
}

pub fn get_queue_ci(properties: Vec<QueueProperties>, instance: &Instance, pdevice: vk::PhysicalDevice, builder: vk::DeviceCreateInfoBuilder<'_>) -> () {
    let max_queue_count = properties.iter().map(|&prop| prop.props.queue_count).max().unwrap();
    let priorities = vec![1.0; max_queue_count as usize];
    
    // Key: queue family index, Value: (requested queue count, actual queue count)
    let queue_map = properties.iter()
        .into_grouping_map_by(|prop| prop.family)
        .fold((0, 0), |acc, _, prop| {
            let request = acc.0 + prop.count;
            (request, request.max(prop.props.queue_count))
        });

    let queue_ci = queue_map
        .iter()
        .map(|(&queue_family_index, (_, queue_count))| {
            vk::DeviceQueueCreateInfo::builder()
                .queue_family_index(queue_family_index)
                .queue_priorities(&priorities[..*queue_count as usize])
                .build() // SAFETY: Priorities is not dropped until end of function
        })
        .collect_vec();

    let device_ci = builder.queue_create_infos(&queue_ci[..]);
    let device = unsafe { instance.create_device(pdevice, &device_ci, None)? };

    let _ = properties.into_iter()
        .into_grouping_map_by(|prop| prop.family)
        .fold(0, |acc, family, prop| {

        })
}