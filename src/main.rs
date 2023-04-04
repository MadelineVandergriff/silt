use silt::loader::*;
use silt::sync::{QueueRequest, QueueType};

/*fn main() {
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
    std::thread::sleep(std::time::Duration::from_secs(1));
}*/

fn main() {
    let silt = silt::model_loading::VulkanData::new();
    silt.run();
}