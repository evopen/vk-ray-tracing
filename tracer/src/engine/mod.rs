mod shaders;
use shaders::Shaders;

use rust_embed::RustEmbed;

use std::{
    borrow::Cow,
    collections::{BTreeMap, LinkedList},
    ffi::{CStr, CString},
    io::Write,
    mem::size_of_val,
    path::Path,
    time::Duration,
};

use bytemuck::{Pod, Zeroable};

use anyhow::{Context, Result};

use ash::version::InstanceV1_1;
use ash::{
    extensions::khr::AccelerationStructure,
    extensions::khr::DeferredHostOperations,
    extensions::khr::RayTracingPipeline,
    extensions::khr::Surface,
    extensions::khr::TimelineSemaphore,
    version::{DeviceV1_0, DeviceV1_2, EntryV1_0, InstanceV1_0, InstanceV1_2},
};
use ash::{
    extensions::{ext::DebugUtils, khr::Swapchain},
    vk, Device, Entry, Instance,
};
use bytemuck::cast_slice;
use log::info;
use vk::{AccelerationStructureInfoNV, AccelerationStructureNV, Handle};
use vk_mem::{AllocatorCreateFlags, MemoryUsage};

const VERTICES: [f32; 9] = [0.25, 0.25, 0.0, 0.75, 0.25, 0.0, 0.50, 0.75, 0.0];
const INDICES: [u32; 3] = [0, 1, 2];

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

    println!(
        "{:?}:\n{:?} [{} ({})] : {}\n",
        message_severity,
        message_type,
        message_id_name,
        &message_id_number.to_string(),
        message,
    );

    vk::FALSE
}

pub struct Engine {
    size: winit::dpi::PhysicalSize<u32>,
    entry: Entry,
    allocator: vk_mem::Allocator,
    device: Device,
    instance: Instance,
    vertices_buffer: vk::Buffer,
    indices_buffer: vk::Buffer,
    transform_buffer: vk::Buffer,
    ray_tracing_ext: RayTracingPipeline,
    ray_tracing_pipeline: vk::Pipeline,
    acceleration_structure: AccelerationStructure,
    bottom_as: vk::AccelerationStructureKHR,
    command_pool: vk::CommandPool,
    queue: vk::Queue,
    ray_gen_sbt_buffer: vk::Buffer,
    hit_sbt_buffer: vk::Buffer,
    miss_sbt_buffer: vk::Buffer,
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
            extension_names_raw.push(DebugUtils::name().as_ptr());

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

            let debug_utils_loader = DebugUtils::new(&entry, &instance);
            let debug_call_back = debug_utils_loader
                .create_debug_utils_messenger(&debug_info, None)
                .unwrap();
            let surface = ash_window::create_surface(&entry, &instance, window, None).unwrap();
            let pdevices = instance
                .enumerate_physical_devices()
                .expect("Physical device error");
            let surface_loader = Surface::new(&entry, &instance);
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
                Swapchain::name().as_ptr(),
                AccelerationStructure::name().as_ptr(),
                RayTracingPipeline::name().as_ptr(),
                DeferredHostOperations::name().as_ptr(),
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

            let queue = device.get_device_queue(queue_family_index as u32, 0);

            let surface_format = surface_loader
                .get_physical_device_surface_formats(pdevice, surface)
                .unwrap()[0];

            let surface_capabilities =
                surface_loader.get_physical_device_surface_capabilities(pdevice, surface)?;
            let mut desired_image_count = surface_capabilities.min_image_count + 1;
            if surface_capabilities.max_image_count > 0
                && desired_image_count > surface_capabilities.max_image_count
            {
                desired_image_count = surface_capabilities.max_image_count;
            }
            let surface_resolution = match surface_capabilities.current_extent.width {
                std::u32::MAX => vk::Extent2D {
                    width: size.width,
                    height: size.height,
                },
                _ => surface_capabilities.current_extent,
            };
            let pre_transform = if surface_capabilities
                .supported_transforms
                .contains(vk::SurfaceTransformFlagsKHR::IDENTITY)
            {
                vk::SurfaceTransformFlagsKHR::IDENTITY
            } else {
                surface_capabilities.current_transform
            };
            let present_modes =
                surface_loader.get_physical_device_surface_present_modes(pdevice, surface)?;
            let present_mode = present_modes
                .iter()
                .cloned()
                .find(|&mode| mode == vk::PresentModeKHR::MAILBOX)
                .unwrap_or(vk::PresentModeKHR::FIFO);
            let swapchain_loader = Swapchain::new(&instance, &device);

            let swapchain_create_info = vk::SwapchainCreateInfoKHR::builder()
                .surface(surface)
                .min_image_count(desired_image_count)
                .image_color_space(surface_format.color_space)
                .image_format(surface_format.format)
                .image_extent(surface_resolution)
                .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
                .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
                .pre_transform(pre_transform)
                .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
                .present_mode(present_mode)
                .clipped(true)
                .image_array_layers(1);

            let swapchain = swapchain_loader.create_swapchain(&swapchain_create_info, None)?;

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

            let present_images = swapchain_loader.get_swapchain_images(swapchain).unwrap();
            let present_image_views: Vec<vk::ImageView> = present_images
                .iter()
                .map(|&image| {
                    let create_view_info = vk::ImageViewCreateInfo::builder()
                        .view_type(vk::ImageViewType::TYPE_2D)
                        .format(surface_format.format)
                        .components(vk::ComponentMapping {
                            r: vk::ComponentSwizzle::R,
                            g: vk::ComponentSwizzle::G,
                            b: vk::ComponentSwizzle::B,
                            a: vk::ComponentSwizzle::A,
                        })
                        .subresource_range(vk::ImageSubresourceRange {
                            aspect_mask: vk::ImageAspectFlags::COLOR,
                            base_mip_level: 0,
                            level_count: 1,
                            base_array_layer: 0,
                            layer_count: 1,
                        })
                        .image(image);
                    device.create_image_view(&create_view_info, None).unwrap()
                })
                .collect();
            let device_memory_properties = instance.get_physical_device_memory_properties(pdevice);

            let ray_tracing_pipeline = RayTracingPipeline::new(&instance, &device);
            let acceleration_structure = AccelerationStructure::new(&instance, &device);
            let mut allocator = vk_mem::Allocator::new(&vk_mem::AllocatorCreateInfo {
                physical_device: pdevice.clone(),
                device: device.clone(),
                instance: instance.clone(),
                flags: vk_mem::AllocatorCreateFlags::from_bits_unchecked(0x0000_0020),
                ..Default::default()
            })?;

            let (vertices_buffer, allocation, info) = allocator.create_buffer(
                &vk::BufferCreateInfo::builder()
                    .size(std::mem::size_of_val(&VERTICES) as u64)
                    .usage(
                        vk::BufferUsageFlags::VERTEX_BUFFER
                            | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
                    )
                    .build(),
                &vk_mem::AllocationCreateInfo::default(),
            )?;

            let (indices_buffer, _, _) = allocator.create_buffer(
                &vk::BufferCreateInfo::builder()
                    .size(std::mem::size_of_val(&INDICES) as u64)
                    .usage(
                        vk::BufferUsageFlags::INDEX_BUFFER
                            | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
                    )
                    .build(),
                &vk_mem::AllocationCreateInfo::default(),
            )?;

            let transform_matrix = glam::Mat4::identity();

            let (transform_buffer, _, _) = allocator.create_buffer(
                &vk::BufferCreateInfo::builder()
                    .size(std::mem::size_of_val(&transform_matrix) as u64)
                    .usage(
                        vk::BufferUsageFlags::VERTEX_BUFFER
                            | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
                    ),
                &vk_mem::AllocationCreateInfo::default(),
            )?;

            Ok(Self {
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
                ray_gen_sbt_buffer: vk::Buffer::null(),
                hit_sbt_buffer: vk::Buffer::null(),
                ray_tracing_ext: ray_tracing_pipeline,
                acceleration_structure,
                ray_tracing_pipeline: vk::Pipeline::null(),
                bottom_as: vk::AccelerationStructureKHR::null(),
                miss_sbt_buffer: vk::Buffer::null(),
            })
        }
    }

    pub fn init(&mut self) -> Result<()> {
        self.create_bottom_level_acceleration_structure()?;
        self.create_ray_tracing_pipeline()?;
        self.create_shader_binding_table()?;
        Ok(())
    }

    fn create_bottom_level_acceleration_structure(&mut self) -> Result<()> {
        let vertex_buffer_device_address = vk::DeviceOrHostAddressConstKHR {
            device_address: self.get_buffer_device_address(self.vertices_buffer),
        };
        let index_buffer_device_address = vk::DeviceOrHostAddressConstKHR {
            device_address: self.get_buffer_device_address(self.indices_buffer),
        };
        let transform_buffer_device_address = vk::DeviceOrHostAddressConstKHR {
            device_address: self.get_buffer_device_address(self.transform_buffer),
        };

        unsafe {
            let geometry = vk::AccelerationStructureGeometryKHR::builder()
                .geometry_type(vk::GeometryTypeKHR::TRIANGLES)
                .geometry(vk::AccelerationStructureGeometryDataKHR {
                    triangles: vk::AccelerationStructureGeometryTrianglesDataKHR::builder()
                        .vertex_data(vertex_buffer_device_address)
                        .index_data(index_buffer_device_address)
                        .transform_data(transform_buffer_device_address)
                        .max_vertex(3)
                        .build(),
                })
                .build();
            let as_build_size = self
                .acceleration_structure
                .get_acceleration_structure_build_sizes(
                    self.device.handle(),
                    vk::AccelerationStructureBuildTypeKHR::DEVICE,
                    &vk::AccelerationStructureBuildGeometryInfoKHR::builder()
                        .ty(vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL)
                        .geometries(&[geometry]),
                    &[1],
                );
            let buffer = self.create_acceleration_structure_buffer(as_build_size)?;

            let as_bottom = self.acceleration_structure.create_acceleration_structure(
                &vk::AccelerationStructureCreateInfoKHR::builder()
                    .buffer(buffer)
                    .size(as_build_size.acceleration_structure_size)
                    .ty(vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL)
                    .build(),
                None,
            )?;

            info!("bottom level as created");
        }

        Ok(())
    }

    fn create_ray_tracing_pipeline(&mut self) -> Result<()> {
        unsafe {
            let bind_group_layout = self.device.create_descriptor_set_layout(
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
            let pipeline_layout = self.device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::builder()
                    .set_layouts(&[bind_group_layout])
                    .build(),
                None,
            )?;

            self.ray_tracing_pipeline = self
                .ray_tracing_ext
                .create_ray_tracing_pipelines(
                    vk::DeferredOperationKHR::null(),
                    vk::PipelineCache::null(),
                    &[vk::RayTracingPipelineCreateInfoKHR::builder()
                        .layout(pipeline_layout)
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
        let mut info = vk::ShaderModuleCreateInfo::builder().build();
        info.p_code = bytemuck::cast_slice(raw_bytes).as_ptr();
        info.code_size = raw_bytes.len();
        unsafe { Ok(self.device.create_shader_module(&info, None)?) }
    }

    fn create_acceleration_structure_buffer(
        &self,
        build_size_info: vk::AccelerationStructureBuildSizesInfoKHR,
    ) -> Result<vk::Buffer> {
        let (buffer, _, _) = self.allocator.create_buffer(
            &vk::BufferCreateInfo::builder()
                .size(build_size_info.acceleration_structure_size)
                .usage(
                    vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR
                        | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
                )
                .build(),
            &vk_mem::AllocationCreateInfo::default(),
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
            self.ray_gen_sbt_buffer = self
                .allocator
                .create_buffer(
                    &vk::BufferCreateInfo::builder()
                        .usage(vk::BufferUsageFlags::SHADER_BINDING_TABLE_KHR)
                        .size(32)
                        .build(),
                    &vk_mem::AllocationCreateInfo::default(),
                )?
                .0;
            let handle = self.ray_tracing_ext.get_ray_tracing_shader_group_handles(
                self.ray_tracing_pipeline,
                0,
                3,
                3 * 32,
            )?;
            dbg!(&handle.len());
            Ok(())
        }
    }

    pub fn input(&self, event: &winit::event::WindowEvent) {}

    pub fn update(&self) -> Result<()> {
        Ok(())
    }

    pub fn render(&self) -> Result<()> {
        unsafe {
            let command_buffer = self
                .device
                .allocate_command_buffers(
                    &vk::CommandBufferAllocateInfo::builder()
                        .command_buffer_count(1)
                        .command_pool(self.command_pool)
                        .build(),
                )?
                .first()
                .unwrap()
                .to_owned();

            self.device.begin_command_buffer(
                command_buffer,
                &vk::CommandBufferBeginInfo::builder().build(),
            );
            self.device.cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::RAY_TRACING_KHR,
                self.ray_tracing_pipeline,
            );
            self.ray_tracing_ext.cmd_trace_rays(
                command_buffer,
                &vk::StridedDeviceAddressRegionKHR::builder().build(),
                &vk::StridedDeviceAddressRegionKHR::builder().build(),
                &vk::StridedDeviceAddressRegionKHR::builder().build(),
                &vk::StridedDeviceAddressRegionKHR::default(),
                800,
                600,
                1,
            );
            self.device.end_command_buffer(command_buffer);

            self.device.queue_submit(
                self.queue,
                &[vk::SubmitInfo::builder().build()],
                vk::Fence::null(),
            )?;

            info!("frame");
        }
        Ok(())
    }
}
