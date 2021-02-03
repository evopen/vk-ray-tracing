use std::mem::ManuallyDrop;

use anyhow::{Context, Result};
use ash::{version::DeviceV1_2, vk};
use log::debug;
use vk_mem::AllocationCreateInfo;

pub struct Buffer {
    pub handle: vk::Buffer,
    allocation: vk_mem::Allocation,
    allocator: ManuallyDrop<vk_mem::Allocator>,
    mapped: bool,
    device_address: Option<vk::DeviceAddress>,
    size: usize,
}

impl Buffer {
    pub fn new<I>(
        size: I,
        buffer_usage: vk::BufferUsageFlags,
        memory_usage: vk_mem::MemoryUsage,
        allocator: vk_mem::Allocator,
    ) -> Result<Self>
    where
        I: num_traits::PrimInt,
    {
        let (handle, allocation, _) = allocator.create_buffer(
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
                    allocator.device.get_buffer_device_address(
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
                allocator: ManuallyDrop::new(allocator),
                mapped: false,
                device_address,
                size: size.to_usize().unwrap(),
            })
        }
    }

    pub fn map(&mut self) -> Result<*mut u8> {
        self.mapped = true;
        Ok(self.allocator.map_memory(&self.allocation)?)
    }

    pub fn unmap(&mut self) {
        self.mapped = false;
        self.allocator.unmap_memory(&self.allocation)
    }

    pub fn device_address(&self) -> Result<vk::DeviceAddress> {
        Ok(self
            .device_address
            .context("buffer does not support device addressing")?)
    }

    pub fn copy_into(&self, ptr: *const u8) -> Result<()> {
        let mapped = self.allocator.map_memory(&self.allocation)?;
        unsafe {
            std::ptr::copy_nonoverlapping(ptr, mapped, self.size);
        }
        self.allocator.unmap_memory(&self.allocation);
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
        self.allocator.destroy_buffer(self.handle, &self.allocation);
    }
}
