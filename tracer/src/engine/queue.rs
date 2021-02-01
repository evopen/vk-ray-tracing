use anyhow::Result;
use ash::{version::DeviceV1_0, vk};
use log::debug;

use super::command_buffer::CommandBuffer;

pub struct Queue {
    handle: vk::Queue,
    device: ash::Device,
}

pub struct Fence {
    handle: vk::Fence,
    device: ash::Device,
}

impl Fence {
    pub fn new(device: &ash::Device, signaled: bool) -> Result<Self> {
        unsafe {
            let device = device.clone();
            let handle = device.create_fence(
                &vk::FenceCreateInfo::builder()
                    .flags(match signaled {
                        true => vk::FenceCreateFlags::SIGNALED,
                        false => vk::FenceCreateFlags::empty(),
                    })
                    .build(),
                None,
            )?;
            Ok(Self { handle, device })
        }
    }

    pub fn wait(&self) -> Result<()> {
        unsafe {
            self.device
                .wait_for_fences(&[self.handle], true, std::u64::MAX)?;
            Ok(())
        }
    }

    pub fn reset(&self) -> Result<()> {
        unsafe {
            self.device.reset_fences(&[self.handle])?;
            Ok(())
        }
    }

    pub fn handle(&self) -> vk::Fence {
        self.handle
    }
}

impl Drop for Fence {
    fn drop(&mut self) {
        self.wait().unwrap();
        unsafe { self.device.destroy_fence(self.handle, None) }
    }
}

impl Queue {
    pub fn new(device: &ash::Device, queue_family_index: u32, queue_index: u32) -> Result<Self> {
        unsafe {
            let device = device.clone();
            let handle = device.get_device_queue(queue_family_index, queue_index);
            Ok(Self { handle, device })
        }
    }

    pub fn submit(
        &self,
        command_buffer: CommandBuffer,
        wait_semaphores: &[vk::Semaphore],
        wait_stages: &[vk::PipelineStageFlags],
        signal_semaphores: &[vk::Semaphore],
    ) -> Result<Fence> {
        unsafe {
            debug!("submitted");
            let fence = Fence::new(&self.device, false)?;

            let mut submit_info = vk::SubmitInfo::builder()
                .command_buffers(&[command_buffer.handle()])
                .wait_semaphores(wait_semaphores)
                .wait_dst_stage_mask(wait_stages)
                .signal_semaphores(signal_semaphores)
                .build();

            self.device
                .queue_submit(self.handle, &[submit_info], fence.handle)?;
            debug!("submitted to device");
            let device = self.device.clone();
            let handle = fence.handle;
            tokio::task::spawn(async move {
                debug!("waiting");
                device
                    .wait_for_fences(&[handle], true, std::u64::MAX)
                    .unwrap();
                drop(command_buffer);
                debug!("freed");
            });

            Ok(fence)
        }
    }

    pub fn handle(&self) -> vk::Queue {
        self.handle
    }
}
