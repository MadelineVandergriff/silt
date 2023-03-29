use silt::model_loading;
use std::time::Instant;
use color_eyre::eyre::Result;
use once_cell::sync::Lazy;

static START_TIME: Lazy<Instant> = Lazy::new(|| Instant::now());

fn main() -> Result<()> {
    Lazy::force(&START_TIME);
    let triangle3 = model_loading::VulkanData::new();
    println!("Elapsed: {:.3}s", START_TIME.elapsed().as_secs_f32());
    triangle3.run();

    Ok(())
}