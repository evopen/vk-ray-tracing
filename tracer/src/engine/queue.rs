use std::{rc::Rc, sync::Arc};

use anyhow::Result;
use ash::{
    version::{DeviceV1_0, DeviceV1_2},
    vk,
};
use log::debug;

use super::{command_buffer::CommandBuffer, Vulkan};

pub struct Queue {
    handle: vk::Queue,
    vulkan: Arc<Vulkan>,
}

pub struct Fence {
    handle: vk::Fence,
    vulkan: Arc<Vulkan>,
}

impl Fence {
    pub fn new(signaled: bool, vulkan: Arc<Vulkan>) -> Result<Self> {
        unsafe {
            let handle = vulkan.device.create_fence(
                &vk::FenceCreateInfo::builder()
                    .flags(match signaled {
                        true => vk::FenceCreateFlags::SIGNALED,
                        false => vk::FenceCreateFlags::empty(),
                    })
                    .build(),
                None,
            )?;
            Ok(Self { handle, vulkan })
        }
    }

    pub fn wait(&self) -> Result<()> {
        unsafe {
            self.vulkan
                .device
                .wait_for_fences(&[self.handle], true, std::u64::MAX)?;
            Ok(())
        }
    }

    pub fn reset(&self) -> Result<()> {
        unsafe {
            self.vulkan.device.reset_fences(&[self.handle])?;
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
        unsafe { self.vulkan.device.destroy_fence(self.handle, None) }
    }
}

pub struct TimelineSemaphore {
    handle: vk::Semaphore,
    device: ash::Device,
}

impl TimelineSemaphore {
    pub fn new(device: &ash::Device) -> Result<Self> {
        unsafe {
            let device = device.clone();
            let handle = device.create_semaphore(
                &vk::SemaphoreCreateInfo::builder()
                    .push_next(
                        &mut vk::SemaphoreTypeCreateInfo::builder()
                            .semaphore_type(vk::SemaphoreType::TIMELINE)
                            .initial_value(0)
                            .build(),
                    )
                    .build(),
                None,
            )?;
            Ok(Self { handle, device })
        }
    }

    pub fn wait_for(&self, value: u64) -> Result<()> {
        unsafe {
            self.device.wait_semaphores(
                &vk::SemaphoreWaitInfo::builder()
                    .semaphores(&[self.handle])
                    .values(&[value])
                    .build(),
                std::u64::MAX,
            )?;
            Ok(())
        }
    }

    pub fn signal(&self, value: u64) -> Result<()> {
        unsafe {
            self.device.signal_semaphore(
                &vk::SemaphoreSignalInfo::builder()
                    .semaphore(self.handle)
                    .value(value)
                    .build(),
            )?;
            Ok(())
        }
    }

    pub fn handle(&self) -> vk::Semaphore {
        self.handle
    }
}

impl Drop for TimelineSemaphore {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_semaphore(self.handle, None);
        }
    }
}

pub struct BinarySemaphore {
    handle: vk::Semaphore,
    device: ash::Device,
}

impl BinarySemaphore {
    pub fn new(device: &ash::Device) -> Result<Self> {
        unsafe {
            let device = device.clone();
            let handle = device.create_semaphore(&vk::SemaphoreCreateInfo::default(), None)?;
            Ok(Self { handle, device })
        }
    }

    pub fn handle(&self) -> vk::Semaphore {
        self.handle
    }
}

impl Drop for BinarySemaphore {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_semaphore(self.handle, None);
        }
    }
}

impl Queue {
    pub fn new(vulkan: Arc<Vulkan>, queue_family_index: u32, queue_index: u32) -> Result<Self> {
        unsafe {
            let handle = vulkan
                .device
                .get_device_queue(queue_family_index, queue_index);
            Ok(Self { handle, vulkan })
        }
    }

    pub fn submit_timeline(
        &self,
        command_buffer: CommandBuffer,
        timeline_semaphores: &[&TimelineSemaphore],
        wait_values: &[u64],
        wait_stages: &[vk::PipelineStageFlags],
        signal_values: &[u64],
    ) -> Result<()> {
        unsafe {
            let semaphore_handles = timeline_semaphores
                .iter()
                .map(|s| s.handle)
                .collect::<Vec<vk::Semaphore>>();

            let mut submit_info = vk::SubmitInfo::builder()
                .command_buffers(&[command_buffer.handle()])
                .wait_semaphores(&semaphore_handles)
                .wait_dst_stage_mask(wait_stages)
                .signal_semaphores(&semaphore_handles)
                .push_next(
                    &mut vk::TimelineSemaphoreSubmitInfo::builder()
                        .wait_semaphore_values(wait_values)
                        .signal_semaphore_values(signal_values)
                        .build(),
                )
                .build();

            let fence = Fence::new(false, self.vulkan.clone())?;
            self.vulkan
                .device
                .queue_submit(self.handle, &[submit_info], fence.handle)?;

            tokio::task::spawn(async move {
                fence.wait().unwrap();
                drop(command_buffer);
            });
            Ok(())
        }
    }

    pub fn submit_binary(
        &self,
        command_buffer: CommandBuffer,
        wait_semaphores: &[vk::Semaphore],
        wait_stages: &[vk::PipelineStageFlags],
        signal_semaphores: &[vk::Semaphore],
    ) -> Result<Arc<Box<Fence>>> {
        unsafe {
            let fence = Arc::new(Box::new(Fence::new(false, self.vulkan.clone())?));

            let mut submit_info = vk::SubmitInfo::builder()
                .command_buffers(&[command_buffer.handle()])
                .wait_semaphores(wait_semaphores)
                .wait_dst_stage_mask(wait_stages)
                .signal_semaphores(signal_semaphores)
                .build();

            self.vulkan
                .device
                .queue_submit(self.handle, &[submit_info], fence.handle)?;
            let device = self.vulkan.device.clone();
            let handle = fence.handle;
            let cmd_buffer_freer = fence.clone();
            tokio::task::spawn(async move {
                device
                    .wait_for_fences(&[cmd_buffer_freer.handle()], true, std::u64::MAX)
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
