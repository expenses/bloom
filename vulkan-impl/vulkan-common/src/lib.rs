use ash::extensions::ext::DebugUtils as DebugUtilsLoader;
use ash::extensions::khr::{Surface as SurfaceLoader, Swapchain as SwapchainLoader};
use ash::vk;
use gpu_allocator::vulkan::{Allocation, AllocationCreateDesc, Allocator};
use std::ffi::{CStr, CString};
use std::ops::Deref;
use std::os::raw::c_char;

// A list of C strings and their associated pointers
pub struct CStrList<'a> {
    list: Vec<&'a CStr>,
    pointers: Vec<*const c_char>,
}

impl<'a> CStrList<'a> {
    pub fn new(list: Vec<&'a CStr>) -> Self {
        let pointers = list.iter().map(|cstr| cstr.as_ptr()).collect();

        Self { list, pointers }
    }

    pub fn pointers(&self) -> &[*const c_char] {
        &self.pointers
    }
}

pub fn select_physical_device(
    instance: &ash::Instance,
    required_extensions: &CStrList,
    surface_loader: &SurfaceLoader,
    surface: vk::SurfaceKHR,
    desired_format: vk::Format,
) -> anyhow::Result<Option<(vk::PhysicalDevice, u32, vk::SurfaceFormatKHR)>> {
    let physical_devices = unsafe { instance.enumerate_physical_devices() }?;

    log::info!(
        "Found {} device{}",
        physical_devices.len(),
        if physical_devices.len() == 1 { "" } else { "s" }
    );

    let selection = physical_devices
        .into_iter()
        .filter_map(|physical_device| unsafe {
            let properties = instance.get_physical_device_properties(physical_device);

            log::info!("");
            log::info!(
                "Checking Device: {:?}",
                cstr_from_array(&properties.device_name)
            );

            log::debug!("Api version: {}", properties.api_version);

            let queue_family = instance
                .get_physical_device_queue_family_properties(physical_device)
                .into_iter()
                .enumerate()
                .position(|(i, queue_family_properties)| {
                    queue_family_properties
                        .queue_flags
                        .contains(vk::QueueFlags::GRAPHICS)
                        && surface_loader
                            .get_physical_device_surface_support(physical_device, i as u32, surface)
                            .unwrap()
                })
                .map(|queue_family| queue_family as u32);

            log::info!(
                "  Checking for a graphics queue family: {}",
                tick(queue_family.is_some())
            );

            let queue_family = match queue_family {
                Some(queue_family) => queue_family,
                None => return None,
            };

            let surface_formats = surface_loader
                .get_physical_device_surface_formats(physical_device, surface)
                .unwrap();

            let surface_format = surface_formats
                .iter()
                .find(|surface_format| {
                    surface_format.format == desired_format
                        && surface_format.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
                })
                .or_else(|| surface_formats.get(0));

            log::info!(
                "  Checking for an appropriate surface format: {}",
                tick(surface_format.is_some())
            );

            let surface_format = match surface_format {
                Some(surface_format) => *surface_format,
                None => return None,
            };

            log::info!("  Checking for required extensions:");

            let supported_device_extensions = instance
                .enumerate_device_extension_properties(physical_device)
                .unwrap();

            let mut has_required_extensions = true;

            for required_extension in &required_extensions.list {
                let device_has_extension = supported_device_extensions.iter().any(|extension| {
                    &cstr_from_array(&extension.extension_name) == required_extension
                });

                log::info!(
                    "    * {:?}: {}",
                    required_extension,
                    tick(device_has_extension)
                );

                has_required_extensions &= device_has_extension;
            }

            if log::log_enabled!(log::Level::Debug) {
                log::debug!("  Supported extensions:");
                supported_device_extensions.iter().for_each(|extension| {
                    log::debug!("    * {:?}", &cstr_from_array(&extension.extension_name));
                });
            }

            if !has_required_extensions {
                return None;
            }

            Some((physical_device, queue_family, surface_format, properties))
        })
        .max_by_key(|(.., properties)| match properties.device_type {
            vk::PhysicalDeviceType::DISCRETE_GPU => 2,
            vk::PhysicalDeviceType::INTEGRATED_GPU => 1,
            _ => 0,
        });

    log::info!("");

    Ok(match selection {
        Some((physical_device, queue_family, surface_format, properties)) => {
            unsafe {
                log::info!(
                    "Using device {:?}",
                    cstr_from_array(&properties.device_name)
                );
            }

            Some((physical_device, queue_family, surface_format))
        }
        None => None,
    })
}

fn tick(supported: bool) -> &'static str {
    if supported {
        "✔️"
    } else {
        "❌"
    }
}

unsafe fn cstr_from_array(array: &[c_char]) -> &CStr {
    CStr::from_ptr(array.as_ptr())
}

pub fn load_shader_module(bytes: &[u8], device: &ash::Device) -> anyhow::Result<vk::ShaderModule> {
    let spv = ash::util::read_spv(&mut std::io::Cursor::new(bytes))?;
    Ok(unsafe {
        device.create_shader_module(&vk::ShaderModuleCreateInfo::builder().code(&spv), None)
    }?)
}

pub fn load_shader_module_as_stage<'a>(
    bytes: &[u8],
    stage: vk::ShaderStageFlags,
    device: &ash::Device,
    entry_point: &'a CStr,
) -> anyhow::Result<vk::PipelineShaderStageCreateInfoBuilder<'a>> {
    let module = load_shader_module(bytes, device)?;

    Ok(vk::PipelineShaderStageCreateInfo::builder()
        .module(module)
        .stage(stage)
        .name(entry_point))
}

pub enum VertexAttribute {
    Float,
    Vec2,
    Vec3,
    Vec4,
}

impl VertexAttribute {
    fn size(&self) -> u32 {
        match self {
            Self::Float => 4,
            Self::Vec2 => 8,
            Self::Vec3 => 12,
            Self::Vec4 => 16,
        }
    }

    fn format(&self) -> vk::Format {
        match self {
            Self::Float => vk::Format::R32_SFLOAT,
            Self::Vec2 => vk::Format::R32G32_SFLOAT,
            Self::Vec3 => vk::Format::R32G32B32_SFLOAT,
            Self::Vec4 => vk::Format::R32G32B32A32_SFLOAT,
        }
    }
}

pub fn create_vertex_attribute_descriptions(
    binding: u32,
    attributes: &[VertexAttribute],
) -> Vec<vk::VertexInputAttributeDescription> {
    let mut descriptions = Vec::with_capacity(attributes.len());

    let mut offset = 0;

    for (i, attribute) in attributes.iter().enumerate() {
        descriptions.push(
            *vk::VertexInputAttributeDescription::builder()
                .binding(binding)
                .location(i as u32)
                .format(attribute.format())
                .offset(offset),
        );

        offset += attribute.size();
    }

    descriptions
}

pub unsafe extern "system" fn vulkan_debug_utils_callback(
    message_severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    message_type: vk::DebugUtilsMessageTypeFlagsEXT,
    p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    _p_user_data: *mut std::ffi::c_void,
) -> vk::Bool32 {
    let filter_out = (message_severity == vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE
        && message_type == vk::DebugUtilsMessageTypeFlagsEXT::GENERAL)
        || (message_severity == vk::DebugUtilsMessageSeverityFlagsEXT::INFO
            && message_type == vk::DebugUtilsMessageTypeFlagsEXT::GENERAL)
        || (message_severity == vk::DebugUtilsMessageSeverityFlagsEXT::INFO
            && message_type == vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION);

    let level = if filter_out {
        log::Level::Trace
    } else {
        match message_severity {
            vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE => log::Level::Debug,
            vk::DebugUtilsMessageSeverityFlagsEXT::INFO => log::Level::Info,
            vk::DebugUtilsMessageSeverityFlagsEXT::WARNING => log::Level::Warn,
            vk::DebugUtilsMessageSeverityFlagsEXT::ERROR => log::Level::Error,
            _ => log::Level::Info,
        }
    };

    let message = std::ffi::CStr::from_ptr((*p_callback_data).p_message);
    let ty = format!("{:?}", message_type).to_lowercase();
    log::log!(level, "[Debug Msg][{}] {:?}", ty, message);
    vk::FALSE
}

pub struct PrimitiveState {
    pub cull_mode: vk::CullModeFlags,
}

pub struct DepthStencilState {
    pub depth_test_enable: bool,
    pub depth_write_enable: bool,
    pub depth_compare_op: vk::CompareOp,
}

pub struct GraphicsPipelineDescriptor<'a> {
    pub primitive_state: PrimitiveState,
    pub depth_stencil_state: DepthStencilState,
    pub vertex_bindings: &'a [vk::VertexInputBindingDescription],
    pub vertex_attributes: &'a [vk::VertexInputAttributeDescription],
    pub colour_attachments: &'a [vk::PipelineColorBlendAttachmentState],
}

impl<'a> GraphicsPipelineDescriptor<'a> {
    pub fn as_baked(&self) -> BakedGraphicsPipelineDescriptor {
        BakedGraphicsPipelineDescriptor {
            input_assembly: vk::PipelineInputAssemblyStateCreateInfo::builder()
                .topology(vk::PrimitiveTopology::TRIANGLE_LIST),
            rasterisation_state: vk::PipelineRasterizationStateCreateInfo::builder()
                .polygon_mode(vk::PolygonMode::FILL)
                .cull_mode(self.primitive_state.cull_mode)
                .line_width(1.0),
            vertex_input: vk::PipelineVertexInputStateCreateInfo::builder()
                .vertex_binding_descriptions(self.vertex_bindings)
                .vertex_attribute_descriptions(self.vertex_attributes),
            viewport_state: vk::PipelineViewportStateCreateInfo::builder()
                .viewport_count(1)
                .scissor_count(1),
            dynamic_state: vk::PipelineDynamicStateCreateInfo::builder()
                .dynamic_states(&[vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR]),
            depth_stencil: vk::PipelineDepthStencilStateCreateInfo::builder()
                .depth_test_enable(self.depth_stencil_state.depth_test_enable)
                .depth_write_enable(self.depth_stencil_state.depth_write_enable)
                .depth_compare_op(self.depth_stencil_state.depth_compare_op),
            multisample_state: vk::PipelineMultisampleStateCreateInfo::builder()
                .sample_shading_enable(false)
                .rasterization_samples(vk::SampleCountFlags::TYPE_1),
            colour_blend_state: vk::PipelineColorBlendStateCreateInfo::builder()
                .logic_op_enable(false)
                .attachments(self.colour_attachments),
        }
    }
}

pub struct BakedGraphicsPipelineDescriptor<'a> {
    pub input_assembly: vk::PipelineInputAssemblyStateCreateInfoBuilder<'a>,
    pub rasterisation_state: vk::PipelineRasterizationStateCreateInfoBuilder<'a>,
    pub vertex_input: vk::PipelineVertexInputStateCreateInfoBuilder<'a>,
    pub viewport_state: vk::PipelineViewportStateCreateInfoBuilder<'a>,
    pub dynamic_state: vk::PipelineDynamicStateCreateInfoBuilder<'a>,
    pub depth_stencil: vk::PipelineDepthStencilStateCreateInfoBuilder<'a>,
    pub multisample_state: vk::PipelineMultisampleStateCreateInfoBuilder<'a>,
    pub colour_blend_state: vk::PipelineColorBlendStateCreateInfoBuilder<'a>,
}

impl<'a> BakedGraphicsPipelineDescriptor<'a> {
    pub fn as_pipeline_create_info(
        &'a self,
        stages: &'a [vk::PipelineShaderStageCreateInfo],
        pipeline_layout: vk::PipelineLayout,
        render_pass: vk::RenderPass,
        subpass: u32,
    ) -> vk::GraphicsPipelineCreateInfoBuilder<'a> {
        vk::GraphicsPipelineCreateInfo::builder()
            .stages(stages)
            .vertex_input_state(&self.vertex_input)
            .input_assembly_state(&self.input_assembly)
            .viewport_state(&self.viewport_state)
            .rasterization_state(&self.rasterisation_state)
            .multisample_state(&self.multisample_state)
            .color_blend_state(&self.colour_blend_state)
            .dynamic_state(&self.dynamic_state)
            .depth_stencil_state(&self.depth_stencil)
            .layout(pipeline_layout)
            .render_pass(render_pass)
            .subpass(subpass)
    }
}

pub fn set_object_name<T: vk::Handle>(
    device: &ash::Device,
    debug_utils_loader: &DebugUtilsLoader,
    handle: T,
    name: &str,
) -> anyhow::Result<()> {
    let name = CString::new(name)?;

    unsafe {
        debug_utils_loader.debug_utils_set_object_name(
            device.handle(),
            &*vk::DebugUtilsObjectNameInfoEXT::builder()
                .object_type(T::TYPE)
                .object_handle(handle.as_raw())
                .object_name(&name),
        )?;
    }

    Ok(())
}

pub struct InitResources<'a> {
    pub command_buffer: vk::CommandBuffer,
    pub device: &'a ash::Device,
    pub allocator: &'a mut Allocator,
    pub debug_utils_loader: Option<&'a DebugUtilsLoader>,
}

pub fn create_image_from_bytes(
    bytes: &[u8],
    extent: vk::Extent3D,
    view_ty: vk::ImageViewType,
    format: vk::Format,
    name: &str,
    init_resources: &mut InitResources,
    next_accesses: &[vk_sync::AccessType],
    next_layout: vk_sync::ImageLayout,
) -> anyhow::Result<(Image, Buffer)> {
    fn ty_from_view_ty(ty: vk::ImageViewType) -> vk::ImageType {
        match ty {
            vk::ImageViewType::TYPE_1D | vk::ImageViewType::TYPE_1D_ARRAY => vk::ImageType::TYPE_1D,
            vk::ImageViewType::TYPE_2D
            | vk::ImageViewType::TYPE_2D_ARRAY
            | vk::ImageViewType::CUBE
            | vk::ImageViewType::CUBE_ARRAY => vk::ImageType::TYPE_2D,
            vk::ImageViewType::TYPE_3D => vk::ImageType::TYPE_3D,
            _ => vk::ImageType::default(),
        }
    }

    let staging_buffer = Buffer::new(
        bytes,
        &format!("{} staging buffer", name),
        vk::BufferUsageFlags::TRANSFER_SRC,
        init_resources,
    )?;

    let image = unsafe {
        init_resources.device.create_image(
            &vk::ImageCreateInfo::builder()
                .image_type(ty_from_view_ty(view_ty))
                .format(format)
                .extent(extent)
                .mip_levels(1)
                .array_layers(1)
                .samples(vk::SampleCountFlags::TYPE_1)
                .usage(vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST),
            None,
        )
    }?;

    let requirements = unsafe { init_resources.device.get_image_memory_requirements(image) };

    let allocation = init_resources.allocator.allocate(&AllocationCreateDesc {
        name,
        requirements,
        location: gpu_allocator::MemoryLocation::GpuOnly,
        linear: false,
    })?;

    unsafe {
        init_resources
            .device
            .bind_image_memory(image, allocation.memory(), allocation.offset())?;
    }

    if let Some(debug_utils_loader) = init_resources.debug_utils_loader {
        set_object_name(&init_resources.device, debug_utils_loader, image, name)?;
    }

    let subresource_range = *vk::ImageSubresourceRange::builder()
        .aspect_mask(vk::ImageAspectFlags::COLOR)
        .level_count(1)
        .layer_count(1);

    let view = unsafe {
        init_resources.device.create_image_view(
            &vk::ImageViewCreateInfo::builder()
                .image(image)
                .view_type(view_ty)
                .format(format)
                .subresource_range(subresource_range),
            None,
        )
    }?;

    unsafe {
        vk_sync::cmd::pipeline_barrier(
            init_resources.device,
            init_resources.command_buffer,
            None,
            &[],
            &[vk_sync::ImageBarrier {
                previous_accesses: &[vk_sync::AccessType::Nothing],
                next_accesses: &[vk_sync::AccessType::TransferWrite],
                next_layout: vk_sync::ImageLayout::Optimal,
                image,
                range: subresource_range,
                discard_contents: true,
                ..Default::default()
            }],
        );

        init_resources.device.cmd_copy_buffer_to_image(
            init_resources.command_buffer,
            staging_buffer.buffer,
            image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            &[*vk::BufferImageCopy::builder()
                .buffer_row_length(extent.width)
                .buffer_image_height(extent.height)
                .image_subresource(vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: 0,
                    base_array_layer: 0,
                    layer_count: 1,
                })
                .image_extent(extent)],
        );

        vk_sync::cmd::pipeline_barrier(
            init_resources.device,
            init_resources.command_buffer,
            None,
            &[],
            &[vk_sync::ImageBarrier {
                previous_accesses: &[vk_sync::AccessType::TransferWrite],
                next_accesses,
                next_layout,
                image,
                range: subresource_range,
                ..Default::default()
            }],
        );
    }

    Ok((
        Image {
            image,
            allocation,
            view,
        },
        staging_buffer,
    ))
}

pub struct Buffer {
    pub allocation: Allocation,
    pub buffer: vk::Buffer,
}

impl Buffer {
    pub fn new(
        bytes: &[u8],
        name: &str,
        usage: vk::BufferUsageFlags,
        init_resources: &mut InitResources,
    ) -> anyhow::Result<Self> {
        let buffer_size = bytes.len() as vk::DeviceSize;

        let buffer = unsafe {
            init_resources.device.create_buffer(
                &vk::BufferCreateInfo::builder()
                    .size(buffer_size)
                    .usage(usage),
                None,
            )
        }?;

        let requirements = unsafe { init_resources.device.get_buffer_memory_requirements(buffer) };

        let allocation = init_resources.allocator.allocate(&AllocationCreateDesc {
            name,
            requirements,
            location: gpu_allocator::MemoryLocation::CpuToGpu,
            linear: true,
        })?;

        Self::from_parts(allocation, buffer, bytes, name, init_resources)
    }

    fn from_parts(
        mut allocation: Allocation,
        buffer: vk::Buffer,
        bytes: &[u8],
        name: &str,
        init_resources: &InitResources,
    ) -> anyhow::Result<Self> {
        let slice = allocation.mapped_slice_mut().unwrap();

        slice[..bytes.len()].copy_from_slice(bytes);

        unsafe {
            init_resources.device.bind_buffer_memory(
                buffer,
                allocation.memory(),
                allocation.offset(),
            )?;
        };

        if let Some(debug_utils_loader) = init_resources.debug_utils_loader {
            set_object_name(&init_resources.device, debug_utils_loader, buffer, name)?;
        }

        Ok(Self { buffer, allocation })
    }

    pub fn cleanup(&self, device: &ash::Device, allocator: &mut Allocator) -> anyhow::Result<()> {
        allocator.free(self.allocation.clone())?;

        unsafe { device.destroy_buffer(self.buffer, None) };

        Ok(())
    }

    // Prefer using this when practical.
    pub fn cleanup_and_drop(
        self,
        device: &ash::Device,
        allocator: &mut Allocator,
    ) -> anyhow::Result<()> {
        self.cleanup(device, allocator)?;
        drop(self);
        Ok(())
    }
}

pub struct ImageDescriptor<'a> {
    pub width: u32,
    pub height: u32,
    pub name: &'a str,
    pub format: vk::Format,
    pub mip_levels: u32,
    pub usage: vk::ImageUsageFlags,
    pub next_accesses: &'a [vk_sync::AccessType],
    pub next_layout: vk_sync::ImageLayout,
}

pub struct Image {
    pub image: vk::Image,
    pub allocation: Allocation,
    pub view: vk::ImageView,
}

impl Image {
    pub fn new(
        descriptor: &ImageDescriptor,
        init_resources: &mut InitResources,
    ) -> anyhow::Result<Self> {
        let &ImageDescriptor {
            width,
            height,
            name,
            format,
            mip_levels,
            usage,
            next_accesses,
            next_layout,
        } = descriptor;

        let image = unsafe {
            init_resources.device.create_image(
                &vk::ImageCreateInfo::builder()
                    .image_type(vk::ImageType::TYPE_2D)
                    .format(format)
                    .extent(vk::Extent3D {
                        width,
                        height,
                        depth: 1,
                    })
                    .mip_levels(mip_levels)
                    .array_layers(1)
                    .samples(vk::SampleCountFlags::TYPE_1)
                    .tiling(vk::ImageTiling::OPTIMAL)
                    .initial_layout(vk::ImageLayout::UNDEFINED)
                    .usage(vk::ImageUsageFlags::TRANSFER_SRC | usage),
                None,
            )
        }?;

        let requirements = unsafe { init_resources.device.get_image_memory_requirements(image) };

        let allocation = init_resources.allocator.allocate(&AllocationCreateDesc {
            name,
            requirements,
            location: gpu_allocator::MemoryLocation::GpuOnly,
            linear: false,
        })?;

        unsafe {
            init_resources.device.bind_image_memory(
                image,
                allocation.memory(),
                allocation.offset(),
            )?;
        };

        if let Some(debug_utils_loader) = init_resources.debug_utils_loader {
            set_object_name(&init_resources.device, debug_utils_loader, image, name)?;
        }

        let subresource_range = *vk::ImageSubresourceRange::builder()
            .aspect_mask(if format == vk::Format::D32_SFLOAT {
                vk::ImageAspectFlags::DEPTH
            } else {
                vk::ImageAspectFlags::COLOR
            })
            .level_count(mip_levels)
            .layer_count(1);

        vk_sync::cmd::pipeline_barrier(
            init_resources.device,
            init_resources.command_buffer,
            None,
            &[],
            &[vk_sync::ImageBarrier {
                previous_accesses: &[vk_sync::AccessType::Nothing],
                next_accesses,
                next_layout,
                image,
                range: subresource_range,
                discard_contents: true,
                ..Default::default()
            }],
        );

        let view = unsafe {
            init_resources.device.create_image_view(
                &vk::ImageViewCreateInfo::builder()
                    .image(image)
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .format(format)
                    .subresource_range(subresource_range),
                None,
            )
        }?;

        Ok(Self {
            image,
            allocation,
            view,
        })
    }

    pub fn cleanup(&self, device: &ash::Device, allocator: &mut Allocator) -> anyhow::Result<()> {
        allocator.free(self.allocation.clone())?;

        unsafe {
            device.destroy_image_view(self.view, None);
            device.destroy_image(self.image, None)
        }

        Ok(())
    }
}

pub struct Swapchain {
    pub swapchain: vk::SwapchainKHR,
    pub images: Vec<vk::Image>,
    pub image_views: Vec<vk::ImageView>,
}

impl Swapchain {
    pub fn new(
        device: &ash::Device,
        swapchain_loader: &SwapchainLoader,
        info: vk::SwapchainCreateInfoKHR,
    ) -> anyhow::Result<Self> {
        unsafe {
            let swapchain = swapchain_loader.create_swapchain(&info, None)?;
            let images = swapchain_loader.get_swapchain_images(swapchain)?;

            /*for (i, image) in images.iter().enumerate() {
                device.set_object_name(*image, &format!("Swapchain image {}", i))?;
            }*/

            let image_views: Vec<_> = images
                .iter()
                .map(|swapchain_image| {
                    let image_view_info = vk::ImageViewCreateInfo::builder()
                        .image(*swapchain_image)
                        .view_type(vk::ImageViewType::TYPE_2D)
                        .format(info.image_format)
                        .subresource_range(
                            vk::ImageSubresourceRange::builder()
                                .aspect_mask(vk::ImageAspectFlags::COLOR)
                                .level_count(1)
                                .layer_count(1)
                                .build(),
                        );
                    device.create_image_view(&image_view_info, None)
                })
                .collect::<Result<Vec<_>, _>>()?;

            Ok(Self {
                images,
                swapchain,
                image_views,
            })
        }
    }
}
