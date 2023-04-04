use crate::vk;

pub struct Image {
    pub image: vk::Image,
    pub view: vk::ImageView,
    pub allocation: vk::Allocation,
}