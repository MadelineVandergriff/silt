use silt::macros::ShaderOptions;
use silt::model::{Vertex, MVP, Model};
use silt::pipeline::{FragmentShader, Shaders, VertexShader};
use silt::prelude::*;
use silt::properties::{DeviceFeatures, DeviceFeaturesRequest, ProvidedFeatures};
use silt::storage::buffer::get_bound_buffer;
use silt::storage::descriptors::{
    get_descriptors, BindingDescription, DescriptorFrequency, DescriptorWriter,
};
use silt::storage::image::{upload_texture, ImageFile, SampledImage};
use silt::sync::{get_command_pools, QueueRequest, QueueType, get_sync_primitives};
use silt::{bindable, shader};
use silt::{loader::*, swapchain::Swapchain};
use silt::{pipeline, storage::descriptors};

use anyhow::Result;

bindable!(
    Texture,
    vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
    DescriptorFrequency::Global,
    1
);

fn main() -> Result<()> {
    let loader_ci = LoaderCreateInfo {
        width: 1920,
        height: 1080,
        title: "Silt Example".into(),
        device_features: DeviceFeaturesRequest {
            required: DeviceFeatures::SAMPLER_ANISOTROPY,
            prefered: DeviceFeatures::IMAGE_CUBE_ARRAY | DeviceFeatures::SPARSE_BINDING,
        },
        queue_requests: vec![QueueRequest {
            ty: QueueType::Graphics,
            count: 1,
        }],
    };

    let (loader, handles) = Loader::new(loader_ci.clone()).unwrap();
    let LoaderHandles {
        debug_messenger,
        surface,
        pdevice,
        queues,
    } = handles;
    let features = ProvidedFeatures::new(&loader, pdevice);

    let present_pass = unsafe { pipeline::get_present_pass(&loader, pdevice, surface) };
    let swapchain = unsafe {
        Swapchain::new(
            &loader,
            surface,
            pdevice,
            present_pass,
            loader_ci.width,
            loader_ci.height,
        )?
    };
    let vertex = VertexShader::new::<Vertex, MVP>(
        &loader,
        shader!("../assets/shaders/model_loading.vert", ShaderOptions::HLSL)?,
    )?;
    let fragment = FragmentShader::new::<Texture>(
        &loader,
        shader!("../assets/shaders/model_loading.frag", ShaderOptions::HLSL)?,
    )?;
    let shaders = Shaders { vertex, fragment };
    let layouts = descriptors::get_layouts(&loader, &[&shaders.vertex, &shaders.fragment])?;
    let pipeline =
        unsafe { pipeline::get_pipeline(&loader, pdevice, present_pass, shaders, &layouts)? };
    let pools = get_command_pools(&loader, &queues, vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)?;

    let texture_image = ImageFile::new("assets/textures/viking_room.png".into())?;
    let texture =
        unsafe { upload_texture::<Texture>(&loader, features, &pools[0], texture_image)? };

    let model = Model::load(&loader, &pools[0], "assets/models/viking_room.obj".into())?;

    let (descriptor_pool, descriptors) = unsafe {
        get_descriptors(
            &loader,
            &layouts,
            &[model.mvp.get_write(), texture.get_write()],
        )?
    };

    let command_buffers = pools[0].get_main_command_buffers(&loader)?;
    let primitives = unsafe { get_sync_primitives(&loader) };

    let record_command_buffer = |parity: Parity, frame: usize| unsafe {
    };

    std::thread::sleep(std::time::Duration::from_secs(1));
    Ok(())
}
