use crate::{loader::Loader, prelude::*};

use anyhow::Result;
use super::QueueHandle;

pub struct CommandPool {
    pub pool: vk::CommandPool,
    pub queue: QueueHandle,
    pub flags: vk::CommandPoolCreateFlags,
}

pub fn get_command_pools(
    loader: &Loader,
    queues: &[QueueHandle],
    flags: vk::CommandPoolCreateFlags,
) -> Result<Vec<CommandPool>> {
    queues
        .iter()
        .map(|queue| {
            let pool_ci = vk::CommandPoolCreateInfo::builder()
                .flags(flags)
                .queue_family_index(queue.family);

            let pool = unsafe { loader.device.create_command_pool(&pool_ci, None)? };

            Ok(CommandPool {
                pool,
                queue: queue.clone(),
                flags,
            })
        })
        .collect::<Result<Vec<_>, _>>()
}
