use silt::macros::ShaderOptions;
use silt::pipeline::{VertexShader, FragmentShader, Shaders};
use silt::storage::descriptors::{BindingDescription, DescriptorFrequency};
use silt::storage::image::SampledImage;
use silt::{loader::*, swapchain::Swapchain};
use silt::model::{MVP, Vertex};
use silt::prelude::*;
use silt::properties::{DeviceFeaturesRequest, DeviceFeatures};
use silt::storage::buffer::{get_bound_buffer};
use silt::sync::{QueueRequest, QueueType};
use silt::{pipeline, storage::descriptors};
use silt::{shader, bindable};

use anyhow::Result;

bindable!(Texture, vk::DescriptorType::SAMPLED_IMAGE, DescriptorFrequency::Global, 1);

fn main() -> Result<()> {
    let loader_ci = LoaderCreateInfo {
        width: 1920,
        height: 1080,
        title: "Silt Example".into(),
        device_features: DeviceFeaturesRequest {
            required: DeviceFeatures::SAMPLER_ANISOTROPY,
            prefered: DeviceFeatures::IMAGE_CUBE_ARRAY | DeviceFeatures::SPARSE_BINDING
        },
        queue_requests: vec![
            QueueRequest { ty: QueueType::Graphics, count: 1 }
        ],
    };

    let (loader, handles) = Loader::new(loader_ci.clone()).unwrap();
    let LoaderHandles { debug_messenger, surface, pdevice, queues } = handles;

    let present_pass = unsafe { pipeline::get_present_pass(&loader, pdevice, surface) };
    let swapchain = unsafe { Swapchain::new(&loader, surface, pdevice, present_pass, loader_ci.width, loader_ci.height) };
    let vertex = VertexShader::new::<Vertex, MVP>(&loader, shader!("../assets/shaders/model_loading.vert", ShaderOptions::HLSL)?)?;
    let fragment = FragmentShader::new::<Texture>(&loader, shader!("../assets/shaders/model_loading.frag", ShaderOptions::HLSL)?)?;
    let shaders = Shaders {
        vertex,
        fragment
    };
    let layouts = descriptors::get_layouts(&loader, &[&shaders.vertex, &shaders.fragment])?;
    let pipeline = unsafe { pipeline::get_pipeline(&loader, pdevice, present_pass, shaders, &layouts)? };

    std::thread::sleep(std::time::Duration::from_secs(1));
    Ok(())
}