mod acceleration_structure;
mod buffer;
mod command_buffer;
mod image;
mod queue;
mod shaders;
mod swapchain;

use acceleration_structure::AccelerationStructure;
use buffer::Buffer;
use command_buffer::CommandBuffer;
use image::Image;
use queue::Queue;
use shaders::Shaders;

use rust_embed::RustEmbed;

use std::{
    borrow::Cow,
    collections::BTreeSet,
    collections::{BTreeMap, LinkedList},
    ffi::{CStr, CString},
    io::Write,
    mem::size_of_val,
    path::Path,
    sync::Arc,
    time::Duration,
};

use bytemuck::{Pod, Zeroable};

use anyhow::{bail, Context, Result};

use ash::{
    extensions::ext,
    extensions::khr,
    version::{DeviceV1_0, DeviceV1_2, EntryV1_0, InstanceV1_0, InstanceV1_1, InstanceV1_2},
};
use ash::{vk, Device, Entry, Instance};
use bytemuck::cast_slice;
use log::{debug, info};
use vk::{
    AccelerationStructureInfoNV, AccelerationStructureNV, Handle, SwapchainKHR,
    WriteDescriptorSetAccelerationStructureKHR,
};
use vk_mem::{AllocatorCreateFlags, MemoryUsage};

use self::{
    queue::{Fence, TimelineSemaphore},
    swapchain::Swapchain,
};

const VERTICES: [f32; 9] = [0.25, 0.25, 0.0, 0.75, 0.25, 0.0, 0.50, 0.75, 0.0];
const INDICES: [u32; 3] = [0, 1, 2];
const TRANSFORM: [f32; 12] = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0];

unsafe extern "system" fn vulkan_debug_callback(
    message_severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    message_type: vk::DebugUtilsMessageTypeFlagsEXT,
    p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    _user_data: *mut std::os::raw::c_void,
) -> vk::Bool32 {
    let callback_data = *p_callback_data;
    let message_id_number: i32 = callback_data.message_id_number as i32;

    let message_id_name = if callback_data.p_message_id_name.is_null() {
        Cow::from("")
    } else {
        CStr::from_ptr(callback_data.p_message_id_name).to_string_lossy()
    };

    let message = if callback_data.p_message.is_null() {
        Cow::from("")
    } else {
        CStr::from_ptr(callback_data.p_message).to_string_lossy()
    };

    if message_severity == vk::DebugUtilsMessageSeverityFlagsEXT::INFO {
        return vk::FALSE;
    }

    println!(
        "{:?}:\n{:?} [{} ({})] : {}\n",
        message_severity,
        message_type,
        message_id_name,
        &message_id_number.to_string(),
        message,
    );

    if message_severity == vk::DebugUtilsMessageSeverityFlagsEXT::ERROR {
        panic!("deal with vulkan validation error, exiting");
    }

    vk::FALSE
}

enum VulkanObject {
    Buffer(vk::Buffer),
    Image(vk::Image),
}

struct Allocation {
    object: VulkanObject,
    allocation: vk_mem::Allocation,
}

pub struct Engine {
    size: winit::dpi::PhysicalSize<u32>,
    entry: Entry,
    device: Device,
    instance: Instance,
    vertices_buffer: Buffer,
    indices_buffer: Buffer,
    transform_buffer: Buffer,
    ray_tracing_pipeline_loader: khr::RayTracingPipeline,
    ray_tracing_pipeline: vk::Pipeline,
    acceleration_structure_loader: khr::AccelerationStructure,
    command_pool: vk::CommandPool,
    queue: Queue,
    ray_gen_sbt_buffer: Option<Buffer>,
    hit_sbt_buffer: Option<Buffer>,
    miss_sbt_buffer: Option<Buffer>,
    descriptor_pool: vk::DescriptorPool,
    descriptor_set_layout: Option<vk::DescriptorSetLayout>,
    descriptor_set: Option<vk::DescriptorSet>,
    allocation_keeper: Vec<Allocation>,
    pipeline_layout: Option<vk::PipelineLayout>,
    swapchain_loader: khr::Swapchain,
    swapchain: Swapchain,
    storage_image: Option<Image>,
    image_layout_keeper: BTreeMap<vk::Image, vk::ImageLayout>,
    ray_tracing_pipeline_properties: vk::PhysicalDeviceRayTracingPipelinePropertiesKHR,
    render_finish_semaphore: vk::Semaphore,
    render_finish_fence: Arc<Box<Fence>>,
    image_available_semaphore: vk::Semaphore,
    instance_buffer: Option<Buffer>,
    bottom_as: Option<AccelerationStructure>,
    top_as: Option<AccelerationStructure>,
    allocator: vk_mem::Allocator,
}

impl Engine {
    pub fn new(window: &winit::window::Window) -> Result<Self> {
        let size = window.inner_size();
        unsafe {
            let entry = Entry::new()?;
            match entry.try_enumerate_instance_version()? {
                // Vulkan 1.1+
                Some(version) => {
                    let major = vk::version_major(version);
                    let minor = vk::version_minor(version);
                    let patch = vk::version_patch(version);
                    info!("{}.{}.{}", major, minor, patch);
                }
                // Vulkan 1.0
                None => {}
            }
            let app_name = CString::new("VulkanTriangle").unwrap();

            let layer_names = [CString::new("VK_LAYER_KHRONOS_validation").unwrap()];
            let layers_names_raw: Vec<*const i8> = layer_names
                .iter()
                .map(|raw_name| raw_name.as_ptr())
                .collect();

            let surface_extensions = ash_window::enumerate_required_extensions(window).unwrap();
            let mut extension_names_raw = surface_extensions
                .iter()
                .map(|ext| ext.as_ptr())
                .collect::<Vec<_>>();
            extension_names_raw.push(ext::DebugUtils::name().as_ptr());

            let appinfo = vk::ApplicationInfo::builder()
                .application_name(&app_name)
                .application_version(0)
                .engine_name(&app_name)
                .engine_version(0)
                .api_version(vk::make_version(1, 2, 0));

            let create_info = vk::InstanceCreateInfo::builder()
                .application_info(&appinfo)
                .enabled_layer_names(&layers_names_raw)
                .enabled_extension_names(&extension_names_raw);

            let instance: Instance = entry
                .create_instance(&create_info, None)
                .expect("Instance creation error");

            let debug_info = vk::DebugUtilsMessengerCreateInfoEXT::builder()
                .message_severity(
                    vk::DebugUtilsMessageSeverityFlagsEXT::ERROR
                        | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
                        | vk::DebugUtilsMessageSeverityFlagsEXT::INFO,
                )
                .message_type(vk::DebugUtilsMessageTypeFlagsEXT::all())
                .pfn_user_callback(Some(vulkan_debug_callback));

            let debug_utils_loader = ext::DebugUtils::new(&entry, &instance);
            let debug_call_back = debug_utils_loader
                .create_debug_utils_messenger(&debug_info, None)
                .unwrap();
            let surface = ash_window::create_surface(&entry, &instance, window, None).unwrap();
            let pdevices = instance
                .enumerate_physical_devices()
                .expect("Physical device error");
            let surface_loader = khr::Surface::new(&entry, &instance);
            let (pdevice, queue_family_index) = pdevices
                .iter()
                .map(|pdevice| {
                    let prop = instance.get_physical_device_properties(*pdevice);

                    instance
                        .get_physical_device_queue_family_properties(*pdevice)
                        .iter()
                        .enumerate()
                        .filter_map(|(index, ref info)| {
                            let supports_graphic_and_surface =
                                info.queue_flags.contains(vk::QueueFlags::GRAPHICS)
                                    && surface_loader
                                        .get_physical_device_surface_support(
                                            *pdevice,
                                            index as u32,
                                            surface,
                                        )
                                        .unwrap();
                            if supports_graphic_and_surface
                                && prop.device_type == vk::PhysicalDeviceType::DISCRETE_GPU
                            {
                                Some((*pdevice, index))
                            } else {
                                None
                            }
                        })
                        .next()
                })
                .filter_map(|v| v)
                .next()
                .expect("Couldn't find suitable device.");

            let queue_family_index = queue_family_index as u32;
            let device_extension_names_raw = [
                khr::Swapchain::name().as_ptr(),
                khr::AccelerationStructure::name().as_ptr(),
                khr::RayTracingPipeline::name().as_ptr(),
                khr::DeferredHostOperations::name().as_ptr(),
                CStr::from_bytes_with_nul(b"VK_KHR_buffer_device_address\0")?.as_ptr(),
                CStr::from_bytes_with_nul(b"VK_EXT_descriptor_indexing\0")?.as_ptr(),
                CStr::from_bytes_with_nul(b"VK_KHR_spirv_1_4\0")?.as_ptr(),
                CStr::from_bytes_with_nul(b"VK_KHR_shader_float_controls\0")?.as_ptr(),
            ];

            let features = vk::PhysicalDeviceFeatures {
                shader_clip_distance: 1,
                ..Default::default()
            };
            let priorities = [1.0];

            let queue_info = [vk::DeviceQueueCreateInfo::builder()
                .queue_family_index(queue_family_index)
                .queue_priorities(&priorities)
                .build()];

            let mut enabled_buffer_device_address_features =
                vk::PhysicalDeviceBufferDeviceAddressFeatures::builder()
                    .buffer_device_address(true)
                    .build();
            let mut enabled_ray_tracing_pipeline_features =
                vk::PhysicalDeviceRayTracingPipelineFeaturesKHR::builder()
                    .ray_tracing_pipeline(true)
                    .build();
            let mut enabled_acceleration_structure_features =
                vk::PhysicalDeviceAccelerationStructureFeaturesKHR::builder()
                    .acceleration_structure(true)
                    .build();

            let device_create_info = vk::DeviceCreateInfo::builder()
                .queue_create_infos(&queue_info)
                .enabled_extension_names(&device_extension_names_raw)
                .enabled_features(&features)
                .push_next(&mut enabled_buffer_device_address_features)
                .push_next(&mut enabled_ray_tracing_pipeline_features)
                .push_next(&mut enabled_acceleration_structure_features)
                .build();

            let device: Device = instance
                .create_device(pdevice, &device_create_info, None)
                .unwrap();

            let queue = Queue::new(&device, queue_family_index as u32, 0)?;

            let swapchain_loader = khr::Swapchain::new(&instance, &device);

            let swapchain = Swapchain::new(
                &swapchain_loader,
                &surface_loader,
                &surface,
                &pdevice,
                &device,
            )?;

            let pool_create_info = vk::CommandPoolCreateInfo::builder()
                .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
                .queue_family_index(queue_family_index);

            let command_pool = device.create_command_pool(&pool_create_info, None)?;

            let command_buffer_allocate_info = vk::CommandBufferAllocateInfo::builder()
                .command_buffer_count(2)
                .command_pool(command_pool)
                .level(vk::CommandBufferLevel::PRIMARY);

            let command_buffers = device.allocate_command_buffers(&command_buffer_allocate_info)?;
            let setup_command_buffer = command_buffers[0];
            let draw_command_buffer = command_buffers[1];

            let mut image_layout_keeper = BTreeMap::new();

            let device_memory_properties = instance.get_physical_device_memory_properties(pdevice);

            let ray_tracing_pipeline = khr::RayTracingPipeline::new(&instance, &device);
            let acceleration_structure_loader = khr::AccelerationStructure::new(&instance, &device);
            let mut allocator = vk_mem::Allocator::new(&vk_mem::AllocatorCreateInfo {
                physical_device: pdevice.clone(),
                device: device.clone(),
                instance: instance.clone(),
                flags: vk_mem::AllocatorCreateFlags::from_bits_unchecked(0x0000_0020),
                ..Default::default()
            })?;
            let mut allocation_keeper = Vec::new();

            let descriptor_pool = device.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::builder()
                    .pool_sizes(&[
                        vk::DescriptorPoolSize::builder()
                            .ty(vk::DescriptorType::STORAGE_IMAGE)
                            .descriptor_count(1)
                            .build(),
                        vk::DescriptorPoolSize::builder()
                            .ty(vk::DescriptorType::ACCELERATION_STRUCTURE_KHR)
                            .descriptor_count(1)
                            .build(),
                    ])
                    .max_sets(1),
                None,
            )?;

            let vertices_buffer = Buffer::new(
                std::mem::size_of_val(&VERTICES),
                vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR
                    | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
                vk_mem::MemoryUsage::CpuToGpu,
                allocator.clone(),
            )?;
            vertices_buffer.copy_into(std::mem::transmute(&VERTICES))?;

            let indices_buffer = Buffer::new(
                std::mem::size_of_val(&INDICES),
                vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR
                    | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
                vk_mem::MemoryUsage::CpuToGpu,
                allocator.clone(),
            )?;
            indices_buffer.copy_into(std::mem::transmute(&INDICES))?;

            let transform_buffer = Buffer::new(
                std::mem::size_of_val(&TRANSFORM),
                vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR
                    | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
                vk_mem::MemoryUsage::CpuToGpu,
                allocator.clone(),
            )?;
            transform_buffer.copy_into(std::mem::transmute(&TRANSFORM))?;

            let render_finish_semaphore =
                device.create_semaphore(&vk::SemaphoreCreateInfo::default(), None)?;
            let image_available_semaphore =
                device.create_semaphore(&vk::SemaphoreCreateInfo::default(), None)?;
            let render_finish_fence = Arc::new(Box::new(Fence::new(&device, true)?));

            let mut ray_tracing_pipeline_properties =
                vk::PhysicalDeviceRayTracingPipelinePropertiesKHR::default();
            instance.get_physical_device_properties2(
                pdevice,
                &mut vk::PhysicalDeviceProperties2::builder()
                    .push_next(&mut ray_tracing_pipeline_properties)
                    .build(),
            );

            Ok(Self {
                ray_tracing_pipeline_properties,
                size,
                entry,
                allocator,
                device,
                instance,
                command_pool,
                vertices_buffer,
                indices_buffer,
                transform_buffer,
                queue,
                ray_gen_sbt_buffer: None,
                hit_sbt_buffer: None,
                ray_tracing_pipeline_loader: ray_tracing_pipeline,
                acceleration_structure_loader,
                ray_tracing_pipeline: vk::Pipeline::null(),
                miss_sbt_buffer: None,
                descriptor_pool,
                descriptor_set_layout: None,
                descriptor_set: None,
                allocation_keeper,
                pipeline_layout: None,
                swapchain_loader,
                swapchain,
                top_as: None,
                storage_image: None,
                image_layout_keeper,
                render_finish_semaphore,
                image_available_semaphore,
                render_finish_fence,
                instance_buffer: None,
                bottom_as: None,
            })
        }
    }

    pub fn init(&mut self) -> Result<()> {
        self.create_storage_image()?;
        info!("storage image created");

        self.create_bottom_level_acceleration_structure()?;
        info!("BLAS created");

        self.create_top_level_acceleration_structure()?;
        info!("TLAS created");

        self.create_ray_tracing_pipeline()?;
        info!("ray tracing pipeline created");

        self.create_descriptor_set()?;
        info!("descriptor set created");

        self.create_shader_binding_table()?;
        info!("binding table created");

        Ok(())
    }

    fn create_storage_image(&mut self) -> Result<()> {
        unsafe {
            let (image, allocation, _) = self.allocator.create_image(
                &vk::ImageCreateInfo::builder()
                    .image_type(vk::ImageType::TYPE_2D)
                    .format(vk::Format::B8G8R8A8_UNORM)
                    .extent(
                        vk::Extent3D::builder()
                            .width(800)
                            .height(600)
                            .depth(1)
                            .build(),
                    )
                    .mip_levels(1)
                    .array_layers(1)
                    .samples(vk::SampleCountFlags::TYPE_1)
                    .tiling(vk::ImageTiling::OPTIMAL)
                    .usage(vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::STORAGE)
                    .initial_layout(vk::ImageLayout::UNDEFINED)
                    .sharing_mode(vk::SharingMode::EXCLUSIVE)
                    .build(),
                &vk_mem::AllocationCreateInfo::default(),
            )?;

            let mut image = Image::new(
                800,
                600,
                vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::STORAGE,
                MemoryUsage::GpuOnly,
                vk::ImageLayout::UNDEFINED,
                &self.allocator,
                &self.device,
            )?;

            let command_buffer = CommandBuffer::new(&self.device, self.command_pool)?;
            command_buffer.encode(|buf| {
                image.cmd_set_layout(buf, vk::ImageLayout::GENERAL)?;
                Ok(())
            });

            self.queue
                .submit_binary(command_buffer, &[], &[], &[])?
                .wait()?;

            self.storage_image = Some(image);
        }
        Ok(())
    }

    fn create_descriptor_set(&mut self) -> Result<()> {
        unsafe {
            self.descriptor_set = Some(
                self.device
                    .allocate_descriptor_sets(
                        &vk::DescriptorSetAllocateInfo::builder()
                            .descriptor_pool(self.descriptor_pool)
                            .set_layouts(&[self.descriptor_set_layout.unwrap()]),
                    )?
                    .first()
                    .unwrap()
                    .to_owned(),
            );
            debug!("descriptor set allocated");

            self.device.update_descriptor_sets(
                &[
                    std::iter::once(
                        vk::WriteDescriptorSet::builder()
                            .descriptor_type(vk::DescriptorType::ACCELERATION_STRUCTURE_KHR)
                            .dst_binding(0)
                            .dst_set(self.descriptor_set.unwrap())
                            .push_next(
                                &mut vk::WriteDescriptorSetAccelerationStructureKHR::builder()
                                    .acceleration_structures(&[self
                                        .top_as
                                        .as_ref()
                                        .unwrap()
                                        .handle()])
                                    .build(),
                            )
                            .build(),
                    )
                    .map(|mut a| {
                        a.descriptor_count = 1;
                        a
                    })
                    .next()
                    .unwrap(),
                    vk::WriteDescriptorSet::builder()
                        .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                        .dst_binding(1)
                        .dst_set(self.descriptor_set.unwrap())
                        .image_info(&[vk::DescriptorImageInfo::builder()
                            .image_view(self.storage_image.as_ref().unwrap().view())
                            .image_layout(vk::ImageLayout::GENERAL)
                            .build()])
                        .build(),
                ],
                &[],
            );
            debug!("descriptor set wrote");
        }
        Ok(())
    }

    fn create_bottom_level_acceleration_structure(&mut self) -> Result<()> {
        let vertex_buffer_device_address = vk::DeviceOrHostAddressConstKHR {
            device_address: self.vertices_buffer.device_address()?,
        };
        let index_buffer_device_address = vk::DeviceOrHostAddressConstKHR {
            device_address: self.indices_buffer.device_address()?,
        };
        let transform_buffer_device_address = vk::DeviceOrHostAddressConstKHR {
            device_address: self.transform_buffer.device_address()?,
        };

        unsafe {
            let geometry = vk::AccelerationStructureGeometryKHR::builder()
                .geometry_type(vk::GeometryTypeKHR::TRIANGLES)
                .flags(vk::GeometryFlagsKHR::OPAQUE)
                .geometry(vk::AccelerationStructureGeometryDataKHR {
                    triangles: vk::AccelerationStructureGeometryTrianglesDataKHR::builder()
                        .vertex_data(vertex_buffer_device_address)
                        .vertex_format(vk::Format::R32G32B32_SFLOAT)
                        .vertex_stride(std::mem::size_of::<f32>() as u64 * 3)
                        .index_data(index_buffer_device_address)
                        .index_type(vk::IndexType::UINT32)
                        .transform_data(transform_buffer_device_address)
                        .max_vertex(3)
                        .build(),
                })
                .build();
            let bottom_as = AccelerationStructure::new(
                &self.device,
                self.command_pool,
                &self.queue,
                &self.acceleration_structure_loader,
                &self.allocator,
                &[geometry],
                vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL,
                1,
            )?;

            self.bottom_as = Some(bottom_as);
        }

        Ok(())
    }

    fn create_top_level_acceleration_structure(&mut self) -> Result<()> {
        unsafe {
            let instance = vk::AccelerationStructureInstanceKHR {
                transform: vk::TransformMatrixKHR { matrix: TRANSFORM },
                instance_custom_index_and_mask: 0 | (0xFF << 24),
                instance_shader_binding_table_record_offset_and_flags: 0 | (0x01 << 24),
                acceleration_structure_reference: vk::AccelerationStructureReferenceKHR {
                    device_handle: self.bottom_as.as_ref().unwrap().device_address(),
                },
            };
            let mut instance_buffer = Buffer::new(
                std::mem::size_of_val(&instance),
                vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                    | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR,
                vk_mem::MemoryUsage::CpuToGpu,
                self.allocator.clone(),
            )?;

            let mapped = instance_buffer.map()?;

            let raw = std::mem::transmute(&instance);
            std::ptr::copy_nonoverlapping(raw, mapped, std::mem::size_of_val(&instance));

            instance_buffer.unmap();

            let geometry = vk::AccelerationStructureGeometryKHR::builder()
                .geometry_type(vk::GeometryTypeKHR::INSTANCES)
                .flags(vk::GeometryFlagsKHR::OPAQUE)
                .geometry(vk::AccelerationStructureGeometryDataKHR {
                    instances: vk::AccelerationStructureGeometryInstancesDataKHR::builder()
                        .array_of_pointers(false)
                        .data(vk::DeviceOrHostAddressConstKHR {
                            device_address: instance_buffer.device_address()?,
                        })
                        .build(),
                })
                .build();

            let top_as = AccelerationStructure::new(
                &self.device,
                self.command_pool,
                &self.queue,
                &self.acceleration_structure_loader,
                &self.allocator,
                &[geometry],
                vk::AccelerationStructureTypeKHR::TOP_LEVEL,
                1,
            )?;

            self.top_as = Some(top_as);
            self.instance_buffer = Some(instance_buffer);
        }
        Ok(())
    }

    fn create_ray_tracing_pipeline(&mut self) -> Result<()> {
        unsafe {
            let descriptor_set_layout = self.device.create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::builder()
                    .bindings(&[
                        vk::DescriptorSetLayoutBinding::builder()
                            .binding(0)
                            .descriptor_count(1)
                            .descriptor_type(vk::DescriptorType::ACCELERATION_STRUCTURE_KHR)
                            .stage_flags(vk::ShaderStageFlags::RAYGEN_KHR)
                            .build(),
                        vk::DescriptorSetLayoutBinding::builder()
                            .binding(1)
                            .descriptor_count(1)
                            .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                            .stage_flags(vk::ShaderStageFlags::RAYGEN_KHR)
                            .build(),
                    ])
                    .build(),
                None,
            )?;
            self.descriptor_set_layout = Some(descriptor_set_layout);
            self.pipeline_layout = Some(
                self.device.create_pipeline_layout(
                    &vk::PipelineLayoutCreateInfo::builder()
                        .set_layouts(&[descriptor_set_layout])
                        .build(),
                    None,
                )?,
            );

            self.ray_tracing_pipeline = self
                .ray_tracing_pipeline_loader
                .create_ray_tracing_pipelines(
                    vk::DeferredOperationKHR::null(),
                    vk::PipelineCache::null(),
                    &[vk::RayTracingPipelineCreateInfoKHR::builder()
                        .max_pipeline_ray_recursion_depth(1)
                        .layout(self.pipeline_layout.unwrap())
                        .stages(&[
                            vk::PipelineShaderStageCreateInfo::builder()
                                .module(self.create_shader_module(
                                    Shaders::get("ray_gen.rgen.spv").unwrap(),
                                )?)
                                .stage(vk::ShaderStageFlags::RAYGEN_KHR)
                                .name(CStr::from_bytes_with_nul(b"main\0")?)
                                .build(),
                            vk::PipelineShaderStageCreateInfo::builder()
                                .module(self.create_shader_module(
                                    Shaders::get("closest_hit.rchit.spv").unwrap(),
                                )?)
                                .stage(vk::ShaderStageFlags::CLOSEST_HIT_KHR)
                                .name(CStr::from_bytes_with_nul(b"main\0")?)
                                .build(),
                            vk::PipelineShaderStageCreateInfo::builder()
                                .module(self.create_shader_module(
                                    Shaders::get("miss.rmiss.spv").unwrap(),
                                )?)
                                .stage(vk::ShaderStageFlags::MISS_KHR)
                                .name(CStr::from_bytes_with_nul(b"main\0")?)
                                .build(),
                        ])
                        .groups(&[
                            vk::RayTracingShaderGroupCreateInfoKHR::builder()
                                .ty(vk::RayTracingShaderGroupTypeKHR::GENERAL)
                                .general_shader(0)
                                .any_hit_shader(vk::SHADER_UNUSED_KHR)
                                .closest_hit_shader(vk::SHADER_UNUSED_KHR)
                                .intersection_shader(vk::SHADER_UNUSED_KHR)
                                .build(),
                            vk::RayTracingShaderGroupCreateInfoKHR::builder()
                                .ty(vk::RayTracingShaderGroupTypeKHR::TRIANGLES_HIT_GROUP)
                                .general_shader(vk::SHADER_UNUSED_KHR)
                                .any_hit_shader(vk::SHADER_UNUSED_KHR)
                                .closest_hit_shader(1)
                                .intersection_shader(vk::SHADER_UNUSED_KHR)
                                .build(),
                            vk::RayTracingShaderGroupCreateInfoKHR::builder()
                                .ty(vk::RayTracingShaderGroupTypeKHR::GENERAL)
                                .general_shader(2)
                                .any_hit_shader(vk::SHADER_UNUSED_KHR)
                                .closest_hit_shader(vk::SHADER_UNUSED_KHR)
                                .intersection_shader(vk::SHADER_UNUSED_KHR)
                                .build(),
                        ])
                        .build()],
                    None,
                )?
                .first()
                .unwrap()
                .to_owned();
        }

        Ok(())
    }

    fn create_shader_module<P>(&self, spv: P) -> Result<vk::ShaderModule>
    where
        P: AsRef<[u8]>,
    {
        let raw_bytes = spv.as_ref();
        let mut info = vk::ShaderModuleCreateInfo::builder()
            .code(bytemuck::cast_slice(raw_bytes))
            .build();

        unsafe { Ok(self.device.create_shader_module(&info, None)?) }
    }

    fn create_acceleration_structure_buffer(
        &mut self,
        build_size_info: vk::AccelerationStructureBuildSizesInfoKHR,
    ) -> Result<Buffer> {
        let buffer = Buffer::new(
            build_size_info.acceleration_structure_size,
            vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            vk_mem::MemoryUsage::CpuToGpu,
            self.allocator.clone(),
        )?;

        Ok(buffer)
    }

    fn get_buffer_device_address(&self, buffer: vk::Buffer) -> u64 {
        unsafe {
            self.device.get_buffer_device_address(
                &vk::BufferDeviceAddressInfo::builder()
                    .buffer(buffer)
                    .build(),
            )
        }
    }

    fn create_shader_binding_table(&mut self) -> Result<()> {
        unsafe {
            let handle_size = self
                .ray_tracing_pipeline_properties
                .shader_group_handle_size as usize;
            let handle_size_aligned = self
                .ray_tracing_pipeline_properties
                .shader_group_handle_alignment as usize;
            let handle = self
                .ray_tracing_pipeline_loader
                .get_ray_tracing_shader_group_handles(
                    self.ray_tracing_pipeline,
                    0,
                    3,
                    3 * handle_size_aligned,
                )?;
            let mut buffer = Buffer::new(
                handle_size,
                vk::BufferUsageFlags::SHADER_BINDING_TABLE_KHR
                    | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
                MemoryUsage::CpuToGpu,
                self.allocator.clone(),
            )?;

            let mapped = buffer.map()?;
            std::ptr::copy_nonoverlapping(handle.as_ptr(), mapped, 32);
            buffer.unmap();
            self.ray_gen_sbt_buffer = Some(buffer);

            let mut buffer = Buffer::new(
                handle_size,
                vk::BufferUsageFlags::SHADER_BINDING_TABLE_KHR
                    | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
                MemoryUsage::CpuToGpu,
                self.allocator.clone(),
            )?;
            let mapped = buffer.map()?;
            std::ptr::copy_nonoverlapping(
                handle.as_ptr().add(handle_size_aligned),
                mapped,
                handle_size,
            );
            buffer.unmap();
            self.hit_sbt_buffer = Some(buffer);

            let mut buffer = Buffer::new(
                handle_size,
                vk::BufferUsageFlags::SHADER_BINDING_TABLE_KHR
                    | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
                MemoryUsage::CpuToGpu,
                self.allocator.clone(),
            )?;
            let mapped = buffer.map()?;
            std::ptr::copy_nonoverlapping(
                handle.as_ptr().add(handle_size_aligned * 2),
                mapped,
                handle_size,
            );
            buffer.unmap();
            self.miss_sbt_buffer = Some(buffer);

            Ok(())
        }
    }

    pub fn input(&self, event: &winit::event::WindowEvent) {}

    pub fn update(&self) -> Result<()> {
        Ok(())
    }

    pub fn render(&mut self) -> Result<()> {
        unsafe {
            let (index, _) = self
                .swapchain
                .acquire_next_image(self.image_available_semaphore)?;
            let handle_size_aligned = self
                .ray_tracing_pipeline_properties
                .shader_group_handle_alignment as u64;

            let command_buffer = CommandBuffer::new(&self.device, self.command_pool)?;

            self.device.begin_command_buffer(
                command_buffer.handle(),
                &vk::CommandBufferBeginInfo::default(),
            )?;

            self.device.cmd_bind_pipeline(
                command_buffer.handle(),
                vk::PipelineBindPoint::RAY_TRACING_KHR,
                self.ray_tracing_pipeline,
            );
            self.device.cmd_bind_descriptor_sets(
                command_buffer.handle(),
                vk::PipelineBindPoint::RAY_TRACING_KHR,
                self.pipeline_layout.unwrap(),
                0,
                &[self.descriptor_set.unwrap()],
                &[],
            );

            self.ray_tracing_pipeline_loader.cmd_trace_rays(
                command_buffer.handle(),
                &vk::StridedDeviceAddressRegionKHR::builder()
                    .device_address(self.ray_gen_sbt_buffer.as_ref().unwrap().device_address()?)
                    .stride(handle_size_aligned)
                    .size(handle_size_aligned)
                    .build(),
                &vk::StridedDeviceAddressRegionKHR::builder()
                    .device_address(self.miss_sbt_buffer.as_ref().unwrap().device_address()?)
                    .stride(handle_size_aligned)
                    .size(handle_size_aligned)
                    .build(),
                &vk::StridedDeviceAddressRegionKHR::builder()
                    .device_address(self.hit_sbt_buffer.as_ref().unwrap().device_address()?)
                    .stride(handle_size_aligned)
                    .size(handle_size_aligned)
                    .build(),
                &vk::StridedDeviceAddressRegionKHR::default(),
                800,
                600,
                1,
            );
            self.storage_image.as_mut().unwrap().cmd_set_layout(
                command_buffer.handle(),
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            );

            self.swapchain.images()[index as usize].cmd_set_layout(
                command_buffer.handle(),
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            )?;

            self.device.cmd_copy_image(
                command_buffer.handle(),
                self.storage_image.as_ref().unwrap().handle(),
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                self.swapchain.images()[index as usize].handle(),
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[vk::ImageCopy::builder()
                    .src_subresource(
                        vk::ImageSubresourceLayers::builder()
                            .layer_count(1)
                            .base_array_layer(0)
                            .aspect_mask(vk::ImageAspectFlags::COLOR)
                            .mip_level(0)
                            .build(),
                    )
                    .dst_subresource(
                        vk::ImageSubresourceLayers::builder()
                            .layer_count(1)
                            .base_array_layer(0)
                            .aspect_mask(vk::ImageAspectFlags::COLOR)
                            .mip_level(0)
                            .build(),
                    )
                    .extent(vk::Extent3D {
                        width: 800,
                        height: 600,
                        depth: 1,
                    })
                    .build()],
            );

            self.swapchain.images()[index as usize]
                .cmd_set_layout(command_buffer.handle(), vk::ImageLayout::PRESENT_SRC_KHR)?;
            self.storage_image
                .as_mut()
                .unwrap()
                .cmd_set_layout(command_buffer.handle(), vk::ImageLayout::GENERAL);
            self.device.end_command_buffer(command_buffer.handle())?;

            debug!("record complete");
            self.render_finish_fence.wait();
            debug!("render finished");

            self.render_finish_fence = self.queue.submit_binary(
                command_buffer,
                &[self.image_available_semaphore],
                &[vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT],
                &[self.render_finish_semaphore],
            )?;

            debug!("command submitted");
            self.swapchain_loader.queue_present(
                self.queue.handle(),
                &vk::PresentInfoKHR::builder()
                    .swapchains(&[self.swapchain.handle()])
                    .image_indices(&[index])
                    .wait_semaphores(&[self.render_finish_semaphore])
                    .build(),
            )?;

            info!("frame presented");
        }
        Ok(())
    }
}

impl Drop for Engine {
    fn drop(&mut self) {
        self.allocation_keeper
            .iter()
            .for_each(|allocation| match allocation.object {
                VulkanObject::Buffer(buffer) => self
                    .allocator
                    .destroy_buffer(buffer, &allocation.allocation),
                VulkanObject::Image(image) => {
                    self.allocator.destroy_image(image, &allocation.allocation)
                }
            });
    }
}
