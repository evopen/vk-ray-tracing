use std::{
    borrow::Cow,
    collections::{BTreeMap, LinkedList},
    ffi::{CStr, CString},
    io::Write,
    mem::size_of_val,
    time::Duration,
};

use bytemuck::{Pod, Zeroable};

use anyhow::{Context, Result};

use ash::{
    extensions::khr::RayTracing,
    extensions::khr::Surface,
    version::{DeviceV1_0, DeviceV1_2, EntryV1_0, InstanceV1_0},
};
use ash::{
    extensions::{ext::DebugUtils, khr::Swapchain},
    vk, Device, Entry, Instance,
};
use bytemuck::cast_slice;
use gpu_allocator::{AllocationCreateDesc, VulkanAllocator};
use log::info;
use vk::AccelerationStructureInfoNV;

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
    allocator: VulkanAllocator,
    device: Device,
    instance: Instance,
    vertices_buffer: vk::Buffer,
    transform_buffer: vk::Buffer,
    ray_tracing: RayTracing,
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
                            if supports_graphic_and_surface {
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
                CStr::from_bytes_with_nul(b"VK_KHR_acceleration_structure\0")
                    .unwrap()
                    .as_ptr(),
                CStr::from_bytes_with_nul(b"VK_KHR_deferred_host_operations\0")
                    .unwrap()
                    .as_ptr(),
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

            let mut enabled_buffer_device_addres_features =
                vk::PhysicalDeviceBufferDeviceAddressFeatures::builder()
                    .buffer_device_address(true)
                    .build();
            let mut enabled_ray_tracing_pipeline_features =
                vk::PhysicalDeviceRayTracingFeaturesKHR::builder()
                    .ray_tracing(true)
                    .build();

            let device_create_info = vk::DeviceCreateInfo::builder()
                .queue_create_infos(&queue_info)
                .enabled_extension_names(&device_extension_names_raw)
                .enabled_features(&features)
                .push_next(&mut enabled_buffer_device_addres_features)
                .push_next(&mut enabled_ray_tracing_pipeline_features);

            let device: Device = instance
                .create_device(pdevice, &device_create_info, None)
                .unwrap();

            let present_queue = device.get_device_queue(queue_family_index as u32, 0);

            let surface_format = surface_loader
                .get_physical_device_surface_formats(pdevice, surface)
                .unwrap()[0];

            let surface_capabilities = surface_loader
                .get_physical_device_surface_capabilities(pdevice, surface)
                .unwrap();
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
            let present_modes = surface_loader
                .get_physical_device_surface_present_modes(pdevice, surface)
                .unwrap();
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

            let swapchain = swapchain_loader
                .create_swapchain(&swapchain_create_info, None)
                .unwrap();

            let pool_create_info = vk::CommandPoolCreateInfo::builder()
                .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
                .queue_family_index(queue_family_index);

            let pool = device.create_command_pool(&pool_create_info, None).unwrap();

            let command_buffer_allocate_info = vk::CommandBufferAllocateInfo::builder()
                .command_buffer_count(2)
                .command_pool(pool)
                .level(vk::CommandBufferLevel::PRIMARY);

            let command_buffers = device
                .allocate_command_buffers(&command_buffer_allocate_info)
                .unwrap();
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

            let mut allocator = VulkanAllocator::new(&gpu_allocator::VulkanAllocatorCreateDesc {
                instance: instance.clone(),
                device: device.clone(),
                physical_device: pdevice,
                debug_settings: Default::default(),
            });

            let vertices_buffer = device.create_buffer(
                &vk::BufferCreateInfo::builder()
                    .size(std::mem::size_of_val(&VERTICES) as u64)
                    .usage(
                        vk::BufferUsageFlags::VERTEX_BUFFER
                            | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
                    )
                    .build(),
                None,
            )?;

            let requirements = device.get_buffer_memory_requirements(vertices_buffer);
            let mut allocation = allocator.allocate(&AllocationCreateDesc {
                name: "vertex",
                requirements,
                location: gpu_allocator::MemoryLocation::CpuToGpu,
                linear: true,
            })?;
            device.bind_buffer_memory(vertices_buffer, allocation.memory(), allocation.offset())?;

            let raw = allocation.mapped_slice_mut().unwrap();
            let bytes = bytemuck::cast_slice(&VERTICES);

            std::ptr::copy_nonoverlapping(bytes.as_ptr(), raw.as_mut_ptr(), bytes.len());

            let transform_matrix = vk::TransformMatrixKHR {
                matrix: [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            };

            let transform_buffer = device.create_buffer(
                &vk::BufferCreateInfo::builder()
                    .size(std::mem::size_of_val(&transform_matrix) as u64)
                    .usage(vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS)
                    .build(),
                None,
            )?;
            let mut allocation = allocator.allocate(&AllocationCreateDesc {
                name: "vertex",
                requirements: device.get_buffer_memory_requirements(transform_buffer),
                location: gpu_allocator::MemoryLocation::CpuToGpu,
                linear: true,
            })?;
            device.bind_buffer_memory(
                transform_buffer,
                allocation.memory(),
                allocation.offset(),
            )?;
            std::ptr::copy_nonoverlapping(
                bytemuck::cast_slice(&transform_matrix.matrix).as_ptr(),
                allocation.mapped_slice_mut().unwrap().as_mut_ptr(),
                std::mem::size_of_val(&transform_matrix),
            );

            let ray_tracing = RayTracing::new(&instance, &device);

            Ok(Self {
                size,
                entry,
                allocator,
                device,
                instance,
                vertices_buffer,
                transform_buffer,
                ray_tracing,
            })
        }
    }

    pub fn init(&mut self) -> Result<()> {
        self.create_bottom_level_acceleration_structure()?;
        Ok(())
    }

    fn create_bottom_level_acceleration_structure(&mut self) -> Result<()> {
        let vertex_buffer_device_address = vk::DeviceOrHostAddressConstKHR {
            device_address: self.get_buffer_device_address(self.vertices_buffer.clone()),
        };
        let mut acceleration_structure_geometry = vk::AccelerationStructureGeometryKHR::builder()
            .geometry_type(vk::GeometryTypeKHR::TRIANGLES)
            .geometry(vk::AccelerationStructureGeometryDataKHR {
                triangles: vk::AccelerationStructureGeometryTrianglesDataKHR::builder()
                    .vertex_format(vk::Format::R32G32B32_SFLOAT)
                    .vertex_data(vertex_buffer_device_address)
                    .vertex_stride(12)
                    .build(),
            })
            .build();
        println!("here");
        unsafe {
            self.ray_tracing.create_acceleration_structure(
                &vk::AccelerationStructureCreateInfoKHR::builder().build(),
                None,
            )?;
        }
        println!("here");

        Ok(())
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

    pub fn input(&self, event: &winit::event::WindowEvent) {}

    pub fn update(&self) -> Result<()> {
        Ok(())
    }

    pub fn render(&self) -> Result<()> {
        Ok(())
    }
}
