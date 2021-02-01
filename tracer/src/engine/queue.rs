use anyhow::Result;
use ash::{version::DeviceV1_0, vk};

pub struct Queue {
    handle: vk::Queue,
}

impl Queue {
    pub fn new(device: &ash::Device, queue_family_index: u32, queue_index: u32) -> Result<Self> {
        unsafe {
            let device = device.clone();
            let handle = device.get_device_queue(queue_family_index, queue_index);
            Ok(Self { handle })
        }
    }

    pub fn handle(&self) -> vk::Queue {
        self.handle
    }
}
