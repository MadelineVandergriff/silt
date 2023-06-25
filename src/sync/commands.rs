use crate::{loader::Loader, prelude::*, resources::ParitySet};

use super::QueueHandle;
use anyhow::Result;

pub struct CommandPool {
    pub pool: vk::CommandPool,
    pub queue: QueueHandle,
    pub flags: vk::CommandPoolCreateFlags,
}

impl Destructible for CommandPool {
    fn destroy(self, loader: &Loader) {
        self.pool.destroy(loader);
    }
}

impl CommandPool {
    pub fn new(
        loader: &Loader,
        queue: &QueueHandle,
        flags: vk::CommandPoolCreateFlags,
    ) -> Result<CommandPool> {
        let pool_ci = vk::CommandPoolCreateInfo::builder()
            .flags(flags)
            .queue_family_index(queue.family);

        let pool = unsafe { loader.device.create_command_pool(&pool_ci, None)? };

        Ok(CommandPool {
            pool,
            queue: queue.clone(),
            flags,
        })
    }
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

impl CommandPool {
    pub fn execute_one_time_commands<F, R>(&self, loader: &Loader, f: F) -> Result<R>
    where
        F: FnOnce(&Loader, vk::CommandBuffer) -> R,
    {
        unsafe {
            let command_buffer_create_info = vk::CommandBufferAllocateInfo::builder()
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_pool(self.pool)
                .command_buffer_count(1);

            let command_buffer = loader
                .device
                .allocate_command_buffers(&command_buffer_create_info)?[0];

            let begin_info = vk::CommandBufferBeginInfo::builder()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            loader
                .device
                .begin_command_buffer(command_buffer, &begin_info)?;

            let ret = f(loader, command_buffer);

            loader.device.end_command_buffer(command_buffer)?;

            let submit_info =
                vk::SubmitInfo::builder().command_buffers(std::slice::from_ref(&command_buffer));

            loader.device.queue_submit(
                self.queue.queues[0],
                std::slice::from_ref(&submit_info),
                vk::Fence::null(),
            )?;
            loader.device.queue_wait_idle(self.queue.queues[0])?;
            loader
                .device
                .free_command_buffers(self.pool, &[command_buffer]);

            Ok(ret)
        }
    }

    pub fn get_main_command_buffers(
        &self,
        loader: &Loader,
    ) -> Result<ParitySet<vk::CommandBuffer>> {
        let buffer_ci = vk::CommandBufferAllocateInfo::builder()
            .command_pool(self.pool)
            .command_buffer_count(2)
            .level(vk::CommandBufferLevel::PRIMARY);

        unsafe {
            Ok(loader
                .device
                .allocate_command_buffers(&buffer_ci)?
                .into_iter()
                .collect())
        }
    }
}

pub trait Recordable {
    fn record(&self, loader: &Loader, command_buffer: vk::CommandBuffer);
}
