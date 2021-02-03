use std::{rc::Rc, sync::Arc};

use ash::{extensions, vk};

use anyhow::Result;

use super::{image::Image, Vulkan};

pub struct Swapchain {
    handle: vk::SwapchainKHR,
    images: Vec<Image>,
    vulkan: Arc<Vulkan>,
}

impl Swapchain {
    pub fn new(vulkan: Arc<Vulkan>) -> Result<Self> {
        unsafe {
            let surface_capabilities = vulkan
                .surface_loader
                .get_physical_device_surface_capabilities(vulkan.physical_device, vulkan.surface)?;

            let surface_format = vulkan
                .surface_loader
                .get_physical_device_surface_formats(vulkan.physical_device, vulkan.surface)
                .unwrap()[0];

            let swapchain_create_info = vk::SwapchainCreateInfoKHR::builder()
                .surface(vulkan.surface)
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
            let handle = vulkan
                .swapchain_loader
                .create_swapchain(&swapchain_create_info, None)?;
            let width = surface_capabilities.current_extent.width;
            let height = surface_capabilities.current_extent.height;
            let images = vulkan
                .swapchain_loader
                .get_swapchain_images(handle)?
                .into_iter()
                .map(|handle| {
                    Image::from_handle(
                        handle,
                        width,
                        height,
                        vk::ImageLayout::UNDEFINED,
                        vulkan.clone(),
                    )
                    .unwrap()
                })
                .collect();
            Ok(Self {
                handle,
                images,
                vulkan,
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
            Ok(self.vulkan.swapchain_loader.acquire_next_image(
                self.handle,
                0,
                semaphore,
                vk::Fence::null(),
            )?)
        }
    }

    pub fn renew(&mut self) -> Result<()> {
        unsafe {
            self.vulkan
                .swapchain_loader
                .destroy_swapchain(self.handle, None);
            let surface_capabilities = self
                .vulkan
                .surface_loader
                .get_physical_device_surface_capabilities(
                    self.vulkan.physical_device,
                    self.vulkan.surface,
                )?;

            let surface_format = self
                .vulkan
                .surface_loader
                .get_physical_device_surface_formats(
                    self.vulkan.physical_device,
                    self.vulkan.surface,
                )?[0];

            let swapchain_create_info = vk::SwapchainCreateInfoKHR::builder()
                .surface(self.vulkan.surface)
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
            self.handle = self
                .vulkan
                .swapchain_loader
                .create_swapchain(&swapchain_create_info, None)?;
            let width = surface_capabilities.current_extent.width;
            let height = surface_capabilities.current_extent.height;
            self.images = self
                .vulkan
                .swapchain_loader
                .get_swapchain_images(self.handle)?
                .into_iter()
                .map(|handle| {
                    Image::from_handle(
                        handle,
                        width,
                        height,
                        vk::ImageLayout::UNDEFINED,
                        self.vulkan.clone(),
                    )
                    .unwrap()
                })
                .collect();
            Ok(())
        }
    }
}

impl Drop for Swapchain {
    fn drop(&mut self) {
        unsafe {
            self.vulkan
                .swapchain_loader
                .destroy_swapchain(self.handle, None)
        }
    }
}
