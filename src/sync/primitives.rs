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

pub fn get_sync_primitives(loader: &Loader) -> ParitySet<SyncPrimitives> {
    panic!()
}