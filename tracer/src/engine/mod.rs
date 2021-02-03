mod acceleration_structure;
mod buffer;
mod camera;
mod command_buffer;
mod image;
mod queue;
mod shaders;
mod swapchain;

use acceleration_structure::AccelerationStructure;
use buffer::Buffer;
use command_buffer::CommandBuffer;
use image::Image;
use khr::Surface;
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
    rc::Rc,
    sync::Arc,
    time::Duration,
};

use bytemuck::{Pod, Zeroable};

use anyhow::{bail, Context, Result};

use ash::vk;
use ash::{
    extensions::ext,
    extensions::khr,
    version::{DeviceV1_0, DeviceV1_2, EntryV1_0, InstanceV1_0, InstanceV1_1, InstanceV1_2},
};
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

pub struct Vulkan {
    entry: ash::Entry,
    device: ash::Device,
    surface_loader: ash::extensions::khr::Surface,
    swapchain_loader: ash::extensions::khr::Swapchain,
    ray_tracing_pipeline_loader: ash::extensions::khr::RayTracingPipeline,
    acceleration_structure_loader: ash::extensions::khr::AccelerationStructure,
    physical_device: vk::PhysicalDevice,
    command_pool: vk::CommandPool,
    surface: vk::SurfaceKHR,
    allocator: vk_mem::Allocator,
}

pub struct Engine {
    size: winit::dpi::PhysicalSize<u32>,
    vertices_buffer: Buffer,
    indices_buffer: Buffer,
    transform_buffer: Buffer,
    ray_tracing_pipeline: vk::Pipeline,
    command_pool: vk::CommandPool,
    ray_gen_sbt_buffer: Option<Buffer>,
    hit_sbt_buffer: Option<Buffer>,
    miss_sbt_buffer: Option<Buffer>,
    descriptor_pool: vk::DescriptorPool,
    descriptor_set_layout: Option<vk::DescriptorSetLayout>,
    descriptor_set: Option<vk::DescriptorSet>,
    pipeline_layout: Option<vk::PipelineLayout>,
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
    vulkan: Arc<Vulkan>,
    queue: Queue,
}

impl Engine {
    pub fn new(window: &winit::window::Window) -> Result<Self> {
        let size = window.inner_size();
        unsafe {
            let entry = ash::Entry::new()?;
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

            let instance: ash::Instance = entry.create_instance(&create_info, None)?;

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

            let device: ash::Device = instance.create_device(pdevice, &device_create_info, None)?;
            let ray_tracing_pipeline_loader = khr::RayTracingPipeline::new(&instance, &device);
            let acceleration_structure_loader = khr::AccelerationStructure::new(&instance, &device);

            let swapchain_loader = khr::Swapchain::new(&instance, &device);

            let allocator = vk_mem::Allocator::new(&vk_mem::AllocatorCreateInfo {
                physical_device: pdevice.clone(),
                device: device.clone(),
                instance: instance.clone(),
                flags: vk_mem::AllocatorCreateFlags::from_bits_unchecked(0x0000_0020),
                ..Default::default()
            })?;

            let pool_create_info = vk::CommandPoolCreateInfo::builder()
                .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
                .queue_family_index(queue_family_index);
            let command_pool = device.create_command_pool(&pool_create_info, None)?;

            let vulkan = Arc::new(Vulkan {
                entry,
                device,
                surface_loader,
                swapchain_loader,
                ray_tracing_pipeline_loader,
                acceleration_structure_loader,
                physical_device: pdevice,
                command_pool,
                surface,
                allocator,
            });

            let queue = Queue::new(vulkan.clone(), queue_family_index as u32, 0)?;

            let swapchain = Swapchain::new(vulkan.clone())?;

            let command_buffer_allocate_info = vk::CommandBufferAllocateInfo::builder()
                .command_buffer_count(2)
                .command_pool(command_pool)
                .level(vk::CommandBufferLevel::PRIMARY);

            let mut image_layout_keeper = BTreeMap::new();

            let device_memory_properties = instance.get_physical_device_memory_properties(pdevice);

            let descriptor_pool = vulkan.device.create_descriptor_pool(
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
                vulkan.clone(),
            )?;
            vertices_buffer.copy_into(std::mem::transmute(&VERTICES))?;

            let indices_buffer = Buffer::new(
                std::mem::size_of_val(&INDICES),
                vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR
                    | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
                vk_mem::MemoryUsage::CpuToGpu,
                vulkan.clone(),
            )?;
            indices_buffer.copy_into(std::mem::transmute(&INDICES))?;

            let transform_buffer = Buffer::new(
                std::mem::size_of_val(&TRANSFORM),
                vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR
                    | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
                vk_mem::MemoryUsage::CpuToGpu,
                vulkan.clone(),
            )?;
            transform_buffer.copy_into(std::mem::transmute(&TRANSFORM))?;

            let render_finish_semaphore = vulkan
                .device
                .create_semaphore(&vk::SemaphoreCreateInfo::default(), None)?;
            let image_available_semaphore = vulkan
                .device
                .create_semaphore(&vk::SemaphoreCreateInfo::default(), None)?;
            let render_finish_fence = Arc::new(Box::new(Fence::new(true, vulkan.clone())?));

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
                command_pool,
                vertices_buffer,
                indices_buffer,
                transform_buffer,
                ray_gen_sbt_buffer: None,
                hit_sbt_buffer: None,
                ray_tracing_pipeline: vk::Pipeline::null(),
                miss_sbt_buffer: None,
                descriptor_pool,
                descriptor_set_layout: None,
                descriptor_set: None,
                pipeline_layout: None,
                swapchain,
                top_as: None,
                storage_image: None,
                image_layout_keeper,
                render_finish_semaphore,
                image_available_semaphore,
                render_finish_fence,
                instance_buffer: None,
                bottom_as: None,
                vulkan,
                queue,
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
            let mut image = Image::new(
                800,
                600,
                vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::STORAGE,
                MemoryUsage::GpuOnly,
                vk::ImageLayout::UNDEFINED,
                self.vulkan.clone(),
            )?;

            let command_buffer = CommandBuffer::new(&self.vulkan.device, self.command_pool)?;
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
                self.vulkan
                    .device
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

            self.vulkan.device.update_descriptor_sets(
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
                &[geometry],
                vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL,
                1,
                self.vulkan.clone(),
                &self.queue,
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
                self.vulkan.clone(),
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
                &[geometry],
                vk::AccelerationStructureTypeKHR::TOP_LEVEL,
                1,
                self.vulkan.clone(),
                &self.queue,
            )?;

            self.top_as = Some(top_as);
            self.instance_buffer = Some(instance_buffer);
        }
        Ok(())
    }

    fn create_ray_tracing_pipeline(&mut self) -> Result<()> {
        unsafe {
            let descriptor_set_layout = self.vulkan.device.create_descriptor_set_layout(
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
                self.vulkan.device.create_pipeline_layout(
                    &vk::PipelineLayoutCreateInfo::builder()
                        .set_layouts(&[descriptor_set_layout])
                        .build(),
                    None,
                )?,
            );

            self.ray_tracing_pipeline = self
                .vulkan
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

        unsafe { Ok(self.vulkan.device.create_shader_module(&info, None)?) }
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
            self.vulkan.clone(),
        )?;

        Ok(buffer)
    }

    fn get_buffer_device_address(&self, buffer: vk::Buffer) -> u64 {
        unsafe {
            self.vulkan.device.get_buffer_device_address(
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
                .vulkan
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
                self.vulkan.clone(),
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
                self.vulkan.clone(),
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
                self.vulkan.clone(),
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

    pub fn input(&mut self, event: &winit::event::WindowEvent) -> Result<()> {
        match event {
            winit::event::WindowEvent::Resized(_) => {
                self.swapchain.renew()?;
            }
            _ => {}
        }

        Ok(())
    }

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

            let command_buffer = CommandBuffer::new(&self.vulkan.device, self.command_pool)?;

            self.vulkan.device.begin_command_buffer(
                command_buffer.handle(),
                &vk::CommandBufferBeginInfo::default(),
            )?;

            self.vulkan.device.cmd_bind_pipeline(
                command_buffer.handle(),
                vk::PipelineBindPoint::RAY_TRACING_KHR,
                self.ray_tracing_pipeline,
            );
            self.vulkan.device.cmd_bind_descriptor_sets(
                command_buffer.handle(),
                vk::PipelineBindPoint::RAY_TRACING_KHR,
                self.pipeline_layout.unwrap(),
                0,
                &[self.descriptor_set.unwrap()],
                &[],
            );

            self.vulkan.ray_tracing_pipeline_loader.cmd_trace_rays(
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

            self.vulkan.device.cmd_copy_image(
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
            self.vulkan
                .device
                .end_command_buffer(command_buffer.handle())?;

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
            self.vulkan.swapchain_loader.queue_present(
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
