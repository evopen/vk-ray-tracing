use anyhow::Result;
use ash::{version::DeviceV1_0, vk};

pub struct CommandBuffer {
    handle: vk::CommandBuffer,
    device: ash::Device,
    pool: vk::CommandPool,
}

impl CommandBuffer {
    pub fn new(device: &ash::Device, pool: vk::CommandPool) -> Result<Self> {
        unsafe {
            let device = device.clone();
            let handle = device
                .allocate_command_buffers(
                    &vk::CommandBufferAllocateInfo::builder()
                        .command_pool(pool)
                        .command_buffer_count(1)
                        .level(vk::CommandBufferLevel::PRIMARY)
                        .build(),
                )?
                .first()
                .unwrap()
                .to_owned();

            Ok(Self {
                handle,
                device,
                pool,
            })
        }
    }

    pub fn handle(&self) -> vk::CommandBuffer {
        self.handle
    }

    pub fn encode<F>(&self, func: F) -> Result<()>
    where
        F: FnOnce(vk::CommandBuffer) -> Result<()>,
    {
        unsafe {
            self.device
                .begin_command_buffer(self.handle, &vk::CommandBufferBeginInfo::default())?;
            func(self.handle);
            self.device.end_command_buffer(self.handle)?;
            Ok(())
        }
    }

    pub fn begin(&self) -> Result<()> {
        unsafe {
            self.device
                .begin_command_buffer(self.handle, &vk::CommandBufferBeginInfo::default())?;
            Ok(())
        }
    }

    pub fn end(&self) -> Result<()> {
        unsafe {
            self.device.end_command_buffer(self.handle)?;
            Ok(())
        }
    }
}

impl Drop for CommandBuffer {
    fn drop(&mut self) {
        unsafe {
            self.device.free_command_buffers(self.pool, &[self.handle]);
        }
    }
}
