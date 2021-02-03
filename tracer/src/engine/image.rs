use anyhow::{bail, Result};
use std::{mem::ManuallyDrop, rc::Rc, sync::Arc};

use ash::{version::DeviceV1_0, vk};

use super::Vulkan;

enum ImageType {
    Allocated(vk_mem::Allocation),
    Swapchain,
}

pub struct Image {
    handle: vk::Image,
    view: vk::ImageView,
    image_type: ImageType,
    width: u32,
    height: u32,
    layout: vk::ImageLayout,
    vulkan: Arc<Vulkan>,
}

impl Image {
    pub fn new(
        width: u32,
        height: u32,
        image_usage: vk::ImageUsageFlags,
        memory_usage: vk_mem::MemoryUsage,
        initial_layout: vk::ImageLayout,
        vulkan: Arc<Vulkan>,
    ) -> Result<Self> {
        unsafe {
            let (handle, allocation, _) = vulkan.allocator.create_image(
                &vk::ImageCreateInfo::builder()
                    .image_type(vk::ImageType::TYPE_2D)
                    .format(vk::Format::B8G8R8A8_UNORM)
                    .extent(vk::Extent3D {
                        width,
                        height,
                        depth: 1,
                    })
                    .samples(vk::SampleCountFlags::TYPE_1)
                    .mip_levels(1)
                    .array_layers(1)
                    .tiling(vk::ImageTiling::OPTIMAL)
                    .usage(image_usage)
                    .sharing_mode(vk::SharingMode::EXCLUSIVE)
                    .initial_layout(initial_layout)
                    .build(),
                &vk_mem::AllocationCreateInfo {
                    usage: memory_usage,
                    ..Default::default()
                },
            )?;
            let view = vulkan.device.create_image_view(
                &vk::ImageViewCreateInfo::builder()
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .format(vk::Format::B8G8R8A8_UNORM)
                    .subresource_range(
                        vk::ImageSubresourceRange::builder()
                            .aspect_mask(vk::ImageAspectFlags::COLOR)
                            .base_mip_level(0)
                            .level_count(1)
                            .base_array_layer(0)
                            .layer_count(1)
                            .build(),
                    )
                    .image(handle)
                    .build(),
                None,
            )?;

            let image_type = ImageType::Allocated(allocation);

            Ok(Self {
                handle,
                view,
                width,
                height,
                layout: initial_layout,
                image_type,
                vulkan,
            })
        }
    }

    pub fn from_handle(
        handle: vk::Image,
        width: u32,
        height: u32,
        layout: vk::ImageLayout,
        vulkan: Arc<Vulkan>,
    ) -> Result<Self> {
        unsafe {
            let image_type = ImageType::Swapchain;
            let view = vulkan.device.create_image_view(
                &vk::ImageViewCreateInfo::builder()
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .format(vk::Format::B8G8R8A8_UNORM)
                    .subresource_range(
                        vk::ImageSubresourceRange::builder()
                            .aspect_mask(vk::ImageAspectFlags::COLOR)
                            .base_mip_level(0)
                            .level_count(1)
                            .base_array_layer(0)
                            .layer_count(1)
                            .build(),
                    )
                    .image(handle)
                    .build(),
                None,
            )?;
            Ok(Self {
                handle,
                view,
                width,
                height,
                layout,
                image_type,
                vulkan,
            })
        }
    }

    pub fn handle(&self) -> vk::Image {
        self.handle
    }

    pub fn view(&self) -> vk::ImageView {
        self.view
    }

    pub fn cmd_set_layout(
        &mut self,
        command_buffer: vk::CommandBuffer,
        layout: vk::ImageLayout,
    ) -> Result<()> {
        cmd_set_image_layout(
            self.layout,
            command_buffer,
            self.handle,
            layout,
            &self.vulkan.device,
        )?;
        self.layout = layout;
        Ok(())
    }
}

impl Drop for Image {
    fn drop(&mut self) {
        match &self.image_type {
            ImageType::Allocated(allocation) => unsafe {
                self.vulkan
                    .allocator
                    .destroy_image(self.handle, &allocation);
            },
            ImageType::Swapchain => {}
        }
    }
}

fn cmd_set_image_layout(
    old_layout: vk::ImageLayout,
    command_buffer: vk::CommandBuffer,
    image: vk::Image,
    new_layout: vk::ImageLayout,
    device: &ash::Device,
) -> Result<()> {
    use vk::AccessFlags;
    use vk::ImageLayout;
    use vk::PipelineStageFlags;
    unsafe {
        let src_access_mask = match old_layout {
            ImageLayout::UNDEFINED => AccessFlags::default(),
            ImageLayout::GENERAL => AccessFlags::default(),
            ImageLayout::COLOR_ATTACHMENT_OPTIMAL => AccessFlags::COLOR_ATTACHMENT_WRITE,
            ImageLayout::TRANSFER_DST_OPTIMAL => AccessFlags::TRANSFER_WRITE,
            ImageLayout::TRANSFER_SRC_OPTIMAL => AccessFlags::TRANSFER_READ,
            ImageLayout::PRESENT_SRC_KHR => AccessFlags::COLOR_ATTACHMENT_READ,
            _ => {
                bail!("unknown old layout {:?}", old_layout);
            }
        };
        let dst_access_mask = match new_layout {
            ImageLayout::COLOR_ATTACHMENT_OPTIMAL => AccessFlags::COLOR_ATTACHMENT_WRITE,
            ImageLayout::GENERAL => AccessFlags::default(),
            ImageLayout::TRANSFER_SRC_OPTIMAL => AccessFlags::TRANSFER_READ,
            ImageLayout::TRANSFER_DST_OPTIMAL => AccessFlags::TRANSFER_WRITE,
            ImageLayout::PRESENT_SRC_KHR => AccessFlags::COLOR_ATTACHMENT_READ,
            _ => {
                bail!("unknown new layout {:?}", new_layout);
            }
        };
        device.cmd_pipeline_barrier(
            command_buffer,
            vk::PipelineStageFlags::ALL_COMMANDS,
            vk::PipelineStageFlags::ALL_COMMANDS,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[vk::ImageMemoryBarrier::builder()
                .image(image)
                .old_layout(old_layout)
                .new_layout(new_layout)
                .src_access_mask(src_access_mask)
                .dst_access_mask(dst_access_mask)
                .subresource_range(
                    vk::ImageSubresourceRange::builder()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .base_mip_level(0)
                        .level_count(1)
                        .base_array_layer(0)
                        .layer_count(1)
                        .build(),
                )
                .build()],
        );
    }
    Ok(())
}
