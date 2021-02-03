use ash::{extensions, vk};

use anyhow::Result;

use super::image::Image;

pub struct Swapchain {
    handle: vk::SwapchainKHR,
    loader: ash::extensions::khr::Swapchain,
    images: Vec<Image>,
}

impl Swapchain {
    pub fn new(
        loader: &ash::extensions::khr::Swapchain,
        surface_loader: &ash::extensions::khr::Surface,
        surface: &vk::SurfaceKHR,
        pdevice: &vk::PhysicalDevice,
        device: &ash::Device,
    ) -> Result<Self> {
        unsafe {
            let surface = surface.clone();
            let pdevice = pdevice.clone();
            let loader = loader.clone();

            let surface_capabilities =
                surface_loader.get_physical_device_surface_capabilities(pdevice, surface)?;

            let surface_format = surface_loader
                .get_physical_device_surface_formats(pdevice, surface)
                .unwrap()[0];

            let swapchain_create_info = vk::SwapchainCreateInfoKHR::builder()
                .surface(surface)
                .min_image_count(2)
                .image_color_space(surface_format.color_space)
                .image_format(surface_format.format)
                .image_extent(surface_capabilities.current_extent)
                .image_usage(
                    vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_DST,
                )
                .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
                .pre_transform(vk::SurfaceTransformFlagsKHR::IDENTITY)
                .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
                .present_mode(vk::PresentModeKHR::FIFO)
                .clipped(true)
                .image_array_layers(1);
            let handle = loader.create_swapchain(&swapchain_create_info, None)?;
            let width = surface_capabilities.current_extent.width;
            let height = surface_capabilities.current_extent.height;
            let images = loader
                .get_swapchain_images(handle)?
                .into_iter()
                .map(|handle| {
                    Image::from_handle(handle, device, width, height, vk::ImageLayout::UNDEFINED)
                        .unwrap()
                })
                .collect();
            Ok(Self {
                handle,
                loader,
                images,
            })
        }
    }

    pub fn handle(&self) -> vk::SwapchainKHR {
        self.handle
    }

    pub fn images(&mut self) -> &mut [Image] {
        &mut self.images
    }

    pub fn acquire_next_image(&self, semaphore: vk::Semaphore) -> Result<(u32, bool)> {
        unsafe {
            Ok(self
                .loader
                .acquire_next_image(self.handle, 0, semaphore, vk::Fence::null())?)
        }
    }
}

impl Drop for Swapchain {
    fn drop(&mut self) {
        unsafe { self.loader.destroy_swapchain(self.handle, None) }
    }
}
