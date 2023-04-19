use silt::loader::*;
use silt::model::MVP;
use silt::prelude::*;
use silt::properties::{DeviceFeaturesRequest, DeviceFeatures};
use silt::storage::buffer::{get_bound_buffer, destroy_buffer};
use silt::storage::descriptors::UniformWrite;
use silt::sync::{QueueRequest, QueueType};

fn main() {
    let loader_ci = LoaderCreateInfo {
        width: 1920,
        height: 1080,
        title: "Silt Example".into(),
        device_features: DeviceFeaturesRequest {
            required: DeviceFeatures::SAMPLER_ANISOTROPY,
            prefered: DeviceFeatures::IMAGE_CUBE_ARRAY | DeviceFeatures::SPARSE_BINDING
        },
        queue_requests: vec![
            QueueRequest { ty: QueueType::Graphics, count: 2 }
        ],
    };

    let (loader, handles) = Loader::new(loader_ci).unwrap();

    let bound = get_bound_buffer::<MVP>(&loader, vk::BufferUsageFlags::UNIFORM_BUFFER).unwrap();
    bound.update(&loader, |mvp| {
        mvp.model *= glam::Mat4::from_rotation_x(std::f32::consts::FRAC_PI_8);
    });
    destroy_buffer(&loader, bound.buffer);

    std::thread::sleep(std::time::Duration::from_secs(1));
}