use anyhow::Result;
use ash::{version::DeviceV1_0, vk};
use log::debug;

use super::command_buffer::CommandBuffer;

pub struct Queue {
    handle: vk::Queue,
    device: ash::Device,
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
    ) -> Result<vk::Fence> {
        unsafe {
            let fence = self
                .device
                .create_fence(&vk::FenceCreateInfo::default(), None)?;

            let mut submit_info = vk::SubmitInfo::builder()
                .command_buffers(&[command_buffer.handle()])
                .wait_semaphores(wait_semaphores)
                .wait_dst_stage_mask(wait_stages)
                .signal_semaphores(signal_semaphores)
                .build();

            self.device
                .queue_submit(self.handle, &[submit_info], fence)?;

            let device = self.device.clone();
            tokio::task::spawn(async move {
                device
                    .wait_for_fences(&[fence], true, std::u64::MAX)
                    .unwrap();
                drop(command_buffer);
            });

            Ok(fence)
        }
    }

    pub fn handle(&self) -> vk::Queue {
        self.handle
    }
}
