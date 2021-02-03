use std::{mem::ManuallyDrop, rc::Rc, sync::Arc};

use anyhow::Result;
use ash::{extensions::khr, vk};

use super::{buffer::Buffer, command_buffer::CommandBuffer, queue::Queue, Vulkan};

pub struct AccelerationStructure {
    handle: vk::AccelerationStructureKHR,
    as_buffer: Buffer,
    device_address: u64,
}

impl AccelerationStructure {
    pub fn new(
        geometries: &[vk::AccelerationStructureGeometryKHR],
        as_type: vk::AccelerationStructureTypeKHR,
        primitive_count: u32,
        vulkan: Arc<Vulkan>,
        queue: &Queue,
    ) -> Result<Self> {
        unsafe {
            let build_geometry_info = vk::AccelerationStructureBuildGeometryInfoKHR::builder()
                .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
                .ty(as_type)
                .geometries(geometries);
            let size_info = vulkan
                .acceleration_structure_loader
                .get_acceleration_structure_build_sizes(
                    vulkan.device.handle(),
                    vk::AccelerationStructureBuildTypeKHR::DEVICE,
                    &build_geometry_info,
                    &[1],
                );
            let as_buffer = Buffer::new(
                size_info.acceleration_structure_size,
                vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR
                    | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
                vk_mem::MemoryUsage::CpuToGpu,
                vulkan.clone(),
            )?;

            let handle = vulkan
                .acceleration_structure_loader
                .create_acceleration_structure(
                    &vk::AccelerationStructureCreateInfoKHR::builder()
                        .ty(as_type)
                        .buffer(as_buffer.handle)
                        .size(size_info.acceleration_structure_size)
                        .build(),
                    None,
                )?;

            let scratch_buffer = Buffer::new(
                size_info.build_scratch_size,
                vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR
                    | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
                vk_mem::MemoryUsage::GpuOnly,
                vulkan.clone(),
            )?;

            let build_geometry_info = build_geometry_info
                .dst_acceleration_structure(handle)
                .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
                .scratch_data(vk::DeviceOrHostAddressKHR {
                    device_address: scratch_buffer.device_address()?,
                });

            let build_range_info = vk::AccelerationStructureBuildRangeInfoKHR::builder()
                .first_vertex(0)
                .primitive_offset(0)
                .transform_offset(0)
                .primitive_count(primitive_count)
                .build();

            let command_buffer = CommandBuffer::new(&vulkan.device, vulkan.command_pool)?;
            command_buffer.begin()?;
            vulkan
                .acceleration_structure_loader
                .cmd_build_acceleration_structures(
                    command_buffer.handle(),
                    &[build_geometry_info.build()],
                    &[&[build_range_info]],
                );
            command_buffer.end()?;
            queue.submit_binary(command_buffer, &[], &[], &[])?.wait()?;

            let device_address = vulkan
                .acceleration_structure_loader
                .get_acceleration_structure_device_address(
                    vulkan.device.handle(),
                    &vk::AccelerationStructureDeviceAddressInfoKHR::builder()
                        .acceleration_structure(handle)
                        .build(),
                );

            Ok(Self {
                handle,
                as_buffer,
                device_address,
            })
        }
    }

    pub fn device_address(&self) -> u64 {
        self.device_address
    }

    pub fn handle(&self) -> vk::AccelerationStructureKHR {
        self.handle
    }
}
