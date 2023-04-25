use crate::prelude::*;

pub struct SyncPrimitives {
    pub image_available: vk::Semaphore,
    pub render_finished: vk::Semaphore,
    pub in_flight: vk::Fence,
}

impl Destructible for SyncPrimitives {
    fn destroy(self, loader: &Loader) {
        self.image_available.destroy(loader);
        self.render_finished.destroy(loader);
        self.in_flight.destroy(loader);
    }
}

pub unsafe fn get_sync_primitives(loader: &Loader) -> ParitySet<SyncPrimitives> {
    ParitySet::from_fn(|| {
        let image_available = loader
            .device
            .create_semaphore(&vk::SemaphoreCreateInfo::default(), None)
            .unwrap();
        let render_finished = loader
            .device
            .create_semaphore(&vk::SemaphoreCreateInfo::default(), None)
            .unwrap();

        let fence_create_info =
            vk::FenceCreateInfo::builder().flags(vk::FenceCreateFlags::SIGNALED);
        let in_flight = loader
            .device
            .create_fence(&fence_create_info, None)
            .unwrap();

        SyncPrimitives {
            image_available,
            render_finished,
            in_flight,
        }
    })
}
