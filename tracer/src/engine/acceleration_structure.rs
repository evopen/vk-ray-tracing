use std::mem::ManuallyDrop;

use anyhow::Result;
use ash::{extensions::khr, vk};

use super::{buffer::Buffer, command_buffer::CommandBuffer, queue::Queue};

pub struct AccelerationStructure {
    handle: vk::AccelerationStructureKHR,
    as_buffer: Buffer,
    device_address: u64,
}

impl AccelerationStructure {
    pub fn new(
        device: &ash::Device,
        pool: vk::CommandPool,
        queue: &Queue,
        loader: &ash::extensions::khr::AccelerationStructure,
        allocator: &vk_mem::Allocator,
        geometries: &[vk::AccelerationStructureGeometryKHR],
        as_type: vk::AccelerationStructureTypeKHR,
    ) -> Result<Self> {
        unsafe {
            let build_geometry_info = vk::AccelerationStructureBuildGeometryInfoKHR::builder()
                .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
                .ty(as_type)
                .geometries(geometries);
            let size_info = loader.get_acceleration_structure_build_sizes(
                device.handle(),
                vk::AccelerationStructureBuildTypeKHR::DEVICE,
                &build_geometry_info,
                &[1],
            );
            let as_buffer = Buffer::new(
                size_info.acceleration_structure_size,
                vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR
                    | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
                vk_mem::MemoryUsage::CpuToGpu,
                allocator.clone(),
            )?;

            let handle = loader.create_acceleration_structure(
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
                vk_mem::MemoryUsage::CpuToGpu,
                allocator.clone(),
            )?;

            let build_geometry_info = build_geometry_info
                .dst_acceleration_structure(handle)
                .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
                .scratch_data(vk::DeviceOrHostAddressKHR {
                    device_address: scratch_buffer.device_address()?,
                });

            dbg!(&build_geometry_info.clone());

            let build_range_info = vk::AccelerationStructureBuildRangeInfoKHR::builder()
                .primitive_count(1)
                .build();

            let command_buffer = CommandBuffer::new(&device, pool)?;
            command_buffer.begin()?;
            loader.cmd_build_acceleration_structures(
                command_buffer.handle(),
                &[build_geometry_info.build()],
                &[&[build_range_info]],
            );
            command_buffer.end()?;
            queue.submit_binary(command_buffer, &[], &[], &[])?.wait()?;

            let device_address = loader.get_acceleration_structure_device_address(
                device.handle(),
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
