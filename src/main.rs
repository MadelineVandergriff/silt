use silt::model_loading;
use silt::shader;

fn main() {
    let triangle3 = model_loading::VulkanData::new();
    triangle3.run();
}