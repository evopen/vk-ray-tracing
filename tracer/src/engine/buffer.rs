use std::{mem::ManuallyDrop, rc::Rc, sync::Arc};

use anyhow::{Context, Result};
use ash::{version::DeviceV1_2, vk};
use log::debug;
use vk_mem::AllocationCreateInfo;

use super::Vulkan;

pub struct Buffer {
    pub handle: vk::Buffer,
    allocation: vk_mem::Allocation,
    mapped: bool,
    device_address: Option<vk::DeviceAddress>,
    size: usize,
    vulkan: Arc<Vulkan>,
}

impl Buffer {
    pub fn new<I>(
        size: I,
        buffer_usage: vk::BufferUsageFlags,
        memory_usage: vk_mem::MemoryUsage,
        vulkan: Arc<Vulkan>,
    ) -> Result<Self>
    where
        I: num_traits::PrimInt,
    {
        let (handle, allocation, _) = vulkan.allocator.create_buffer(
            &vk::BufferCreateInfo::builder()
                .usage(buffer_usage)
                .size(size.to_u64().context("failed to convert to u64")?)
                .build(),
            &vk_mem::AllocationCreateInfo {
                usage: memory_usage,
                ..Default::default()
            },
        )?;

        unsafe {
            let device_address = match buffer_usage & vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                != vk::BufferUsageFlags::empty()
            {
                true => Some(
                    vulkan.device.get_buffer_device_address(
                        &vk::BufferDeviceAddressInfo::builder()
                            .buffer(handle)
                            .build(),
                    ),
                ),
                false => None,
            };

            Ok(Self {
                handle,
                allocation,
                mapped: false,
                device_address,
                size: size.to_usize().unwrap(),
                vulkan,
            })
        }
    }

    pub fn map(&mut self) -> Result<*mut u8> {
        self.mapped = true;
        Ok(self.vulkan.allocator.map_memory(&self.allocation)?)
    }

    pub fn unmap(&mut self) {
        self.mapped = false;
        self.vulkan.allocator.unmap_memory(&self.allocation)
    }

    pub fn device_address(&self) -> Result<vk::DeviceAddress> {
        Ok(self
            .device_address
            .context("buffer does not support device addressing")?)
    }

    pub fn copy_into(&self, ptr: *const u8) -> Result<()> {
        let mapped = self.vulkan.allocator.map_memory(&self.allocation)?;
        unsafe {
            std::ptr::copy_nonoverlapping(ptr, mapped, self.size);
        }
        self.vulkan.allocator.unmap_memory(&self.allocation);
        Ok(())
    }

    pub fn size(&self) -> usize {
        self.size
    }
}

impl Drop for Buffer {
    fn drop(&mut self) {
        if self.mapped {
            self.unmap();
        }
        self.vulkan
            .allocator
            .destroy_buffer(self.handle, &self.allocation);
    }
}
