use ash::extensions::ext::DebugUtils as DebugUtilsLoader;
use ash::extensions::khr::{Surface as SurfaceLoader, Swapchain as SwapchainLoader};
use ash::vk;
use rendering_shaders::FilterConstants;
use std::ffi::{CStr, CString};
use ultraviolet::{Mat4, Vec2, Vec3};
use vulkan_common::CStrList;
use winit::event::{ElementState, Event, KeyboardInput, VirtualKeyCode, WindowEvent};
use winit::event_loop::ControlFlow;
use winit::window::Fullscreen;

// I get 10 mip levels on a 2560 x 1600 display, so 12 is probably enough even for 4k.
const MAX_MIPS: u32 = 12;

fn main() -> anyhow::Result<()> {
    {
        use simplelog::*;

        CombinedLogger::init(vec![TermLogger::new(
            LevelFilter::Info,
            Config::default(),
            TerminalMode::Mixed,
            ColorChoice::Auto,
        )])?;
    }

    let event_loop = winit::event_loop::EventLoop::new();
    let window = winit::window::Window::new(&event_loop)?;

    let entry = unsafe { ash::Entry::new() }?;

    let api_version = vk::API_VERSION_1_0;

    let app_info = vk::ApplicationInfo::builder()
        .application_name(CStr::from_bytes_with_nul(b"Bloom\0")?)
        .application_version(vk::make_api_version(0, 0, 1, 0))
        .engine_version(vk::make_api_version(0, 0, 1, 0))
        .api_version(api_version);

    let instance_extensions = CStrList::new({
        let mut instance_extensions = ash_window::enumerate_required_extensions(&window)?;
        instance_extensions.push(DebugUtilsLoader::name());
        instance_extensions
    });

    let enabled_layers = CStrList::new(vec![CStr::from_bytes_with_nul(
        b"VK_LAYER_KHRONOS_validation\0",
    )?]);

    let device_extensions = CStrList::new(vec![SwapchainLoader::name()]);

    let mut debug_messenger_info = vk::DebugUtilsMessengerCreateInfoEXT::builder()
        .message_severity(vk::DebugUtilsMessageSeverityFlagsEXT::all())
        .message_type(vk::DebugUtilsMessageTypeFlagsEXT::all())
        .pfn_user_callback(Some(vulkan_common::vulkan_debug_utils_callback));

    let instance = unsafe {
        entry.create_instance(
            &vk::InstanceCreateInfo::builder()
                .application_info(&app_info)
                .enabled_extension_names(instance_extensions.pointers())
                .enabled_layer_names(enabled_layers.pointers())
                .push_next(&mut debug_messenger_info),
            None,
        )
    }?;

    let debug_utils_loader = DebugUtilsLoader::new(&entry, &instance);
    let _debug_messenger =
        unsafe { debug_utils_loader.create_debug_utils_messenger(&debug_messenger_info, None) }?;

    let surface_loader = SurfaceLoader::new(&entry, &instance);

    let surface = unsafe { ash_window::create_surface(&entry, &instance, &window, None) }?;

    let (physical_device, graphics_queue_family, surface_format) =
        match vulkan_common::select_physical_device(
            &instance,
            &device_extensions,
            &surface_loader,
            surface,
            vk::Format::B8G8R8A8_SRGB,
        )? {
            Some(selection) => selection,
            None => {
                log::info!("No suitable device found 💔. Exiting program");
                return Ok(());
            }
        };

    let surface_caps = unsafe {
        surface_loader.get_physical_device_surface_capabilities(physical_device, surface)
    }?;

    let device = {
        let queue_info = [*vk::DeviceQueueCreateInfo::builder()
            .queue_family_index(graphics_queue_family)
            .queue_priorities(&[1.0])];

        let device_features = vk::PhysicalDeviceFeatures::builder();

        let device_info = vk::DeviceCreateInfo::builder()
            .queue_create_infos(&queue_info)
            .enabled_features(&device_features)
            .enabled_extension_names(device_extensions.pointers())
            .enabled_layer_names(enabled_layers.pointers());

        unsafe { instance.create_device(physical_device, &device_info, None) }?
    };

    let pipeline_cache =
        unsafe { device.create_pipeline_cache(&vk::PipelineCacheCreateInfo::default(), None) }?;

    let descriptor_set_layouts = DescriptorSetLayouts::new(&device)?;
    let render_passes = RenderPasses::new(&device, surface_format.format)?;
    let graphics_pipelines = GraphicsPipelines::new(
        &device,
        &descriptor_set_layouts,
        &render_passes,
        pipeline_cache,
    )?;
    let compute_pipelines =
        ComputePipelines::new(&device, &descriptor_set_layouts, pipeline_cache)?;

    let sampler = unsafe {
        device.create_sampler(
            &vk::SamplerCreateInfo::builder()
                .mag_filter(vk::Filter::LINEAR)
                .min_filter(vk::Filter::LINEAR)
                .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                .max_lod(vk::LOD_CLAMP_NONE),
            None,
        )
    }?;

    let queue = unsafe { device.get_device_queue(graphics_queue_family, 0) };

    let (init_command_buffer, init_command_pool) = {
        let command_pool = unsafe {
            device.create_command_pool(
                &vk::CommandPoolCreateInfo::builder().queue_family_index(graphics_queue_family),
                None,
            )
        }?;

        let cmd_buf_allocate_info = vk::CommandBufferAllocateInfo::builder()
            .command_pool(command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);

        let command_buffer = unsafe { device.allocate_command_buffers(&cmd_buf_allocate_info) }?[0];

        (command_buffer, command_pool)
    };

    let mut buffers_to_cleanup = Vec::new();

    let mut allocator =
        gpu_allocator::vulkan::Allocator::new(&gpu_allocator::vulkan::AllocatorCreateDesc {
            instance: instance.clone(),
            device: device.clone(),
            physical_device,
            debug_settings: gpu_allocator::AllocatorDebugSettings::default(),
            buffer_device_address: false,
        })?;

    // Load buffers and textures

    unsafe {
        device.begin_command_buffer(
            init_command_buffer,
            &vk::CommandBufferBeginInfo::builder()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
        )
    }?;

    let mut init_resources = vulkan_common::InitResources {
        command_buffer: init_command_buffer,
        device: &device,
        allocator: &mut allocator,
        debug_utils_loader: Some(&debug_utils_loader),
    };

    let (vertex_buffer, index_buffer, num_indices, texture) = load_gltf_from_bytes(
        include_bytes!("../../bloom_example.glb"),
        &mut init_resources,
        &mut buffers_to_cleanup,
    )?;

    let window_size = window.inner_size();

    let mut extent = vk::Extent2D {
        width: window_size.width,
        height: window_size.height,
    };

    let mut hdr_framebuffer =
        create_hdr_framebuffer(extent.width, extent.height, &mut init_resources)?;

    let mut depthbuffer = create_depthbuffer(extent.width, extent.height, &mut init_resources)?;

    let sampled_images = 3;
    let bloom_textures = 1;
    let storage_textures = 1;

    let descriptor_pool = unsafe {
        device.create_descriptor_pool(
            &vk::DescriptorPoolCreateInfo::builder()
                .pool_sizes(&[
                    *vk::DescriptorPoolSize::builder()
                        .ty(vk::DescriptorType::SAMPLED_IMAGE)
                        .descriptor_count(sampled_images),
                    *vk::DescriptorPoolSize::builder()
                        .ty(vk::DescriptorType::SAMPLER)
                        .descriptor_count(sampled_images),
                    *vk::DescriptorPoolSize::builder()
                        .ty(vk::DescriptorType::STORAGE_IMAGE)
                        .descriptor_count(bloom_textures * MAX_MIPS + storage_textures),
                ])
                .max_sets(sampled_images + bloom_textures + storage_textures),
            None,
        )
    }?;

    let bloom_mips = bloom_mips_for_dimensions(extent.width, extent.height);

    let mut bloom_texture = BloomTextureWithMips::new(
        &mut init_resources,
        extent.width / 2,
        extent.height / 2,
        bloom_mips,
        &descriptor_set_layouts,
        descriptor_pool,
        sampler,
    )?;

    drop(init_resources);

    unsafe {
        device.end_command_buffer(init_command_buffer)?;
        let fence = device.create_fence(&vk::FenceCreateInfo::builder(), None)?;

        device.queue_submit(
            queue,
            &[*vk::SubmitInfo::builder().command_buffers(&[init_command_buffer])],
            fence,
        )?;

        device.wait_for_fences(&[fence], true, u64::MAX)?;
    }

    for buffers in buffers_to_cleanup.drain(..) {
        buffers.cleanup_and_drop(&device, &mut allocator)?;
    }

    // Bind descriptor sets

    let descriptor_sets = unsafe {
        device.allocate_descriptor_sets(
            &vk::DescriptorSetAllocateInfo::builder()
                .set_layouts(&[
                    descriptor_set_layouts.sampled_texture,
                    descriptor_set_layouts.sampled_texture,
                    descriptor_set_layouts.storage_texture,
                ])
                .descriptor_pool(descriptor_pool),
        )
    }?;

    let descriptor_set = descriptor_sets[0];
    let tonemap_descriptor_set = descriptor_sets[1];
    let hdr_framebuffer_storage_ds = descriptor_sets[2];

    unsafe {
        device.update_descriptor_sets(
            &[
                *vk::WriteDescriptorSet::builder()
                    .dst_set(descriptor_set)
                    .dst_binding(0)
                    .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                    .image_info(&[*vk::DescriptorImageInfo::builder()
                        .image_view(texture.view)
                        .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)]),
                *vk::WriteDescriptorSet::builder()
                    .dst_set(descriptor_set)
                    .dst_binding(1)
                    .descriptor_type(vk::DescriptorType::SAMPLER)
                    .image_info(&[*vk::DescriptorImageInfo::builder().sampler(sampler)]),
                *vk::WriteDescriptorSet::builder()
                    .dst_set(tonemap_descriptor_set)
                    .dst_binding(0)
                    .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                    .image_info(&[*vk::DescriptorImageInfo::builder()
                        .image_view(hdr_framebuffer.view)
                        .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)]),
                *vk::WriteDescriptorSet::builder()
                    .dst_set(tonemap_descriptor_set)
                    .dst_binding(1)
                    .descriptor_type(vk::DescriptorType::SAMPLER)
                    .image_info(&[*vk::DescriptorImageInfo::builder().sampler(sampler)]),
                *vk::WriteDescriptorSet::builder()
                    .dst_set(hdr_framebuffer_storage_ds)
                    .dst_binding(0)
                    .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                    .image_info(&[*vk::DescriptorImageInfo::builder()
                        .image_view(hdr_framebuffer.view)
                        .image_layout(vk::ImageLayout::GENERAL)]),
            ],
            &[],
        );
    }

    // Swapchain

    let mut image_count = (surface_caps.min_image_count + 1).max(3);
    if surface_caps.max_image_count > 0 && image_count > surface_caps.max_image_count {
        image_count = surface_caps.max_image_count;
    }

    log::info!("Using {} swapchain images at a time.", image_count);

    let swapchain_loader = SwapchainLoader::new(&instance, &device);

    let mut swapchain_info = *vk::SwapchainCreateInfoKHR::builder()
        .surface(surface)
        .min_image_count(image_count)
        .image_format(surface_format.format)
        .image_color_space(surface_format.color_space)
        .image_extent(extent)
        .image_array_layers(1)
        .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
        .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
        .pre_transform(surface_caps.current_transform)
        .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
        .present_mode(vk::PresentModeKHR::FIFO)
        .clipped(true)
        .old_swapchain(vk::SwapchainKHR::null());

    let mut swapchain = vulkan_common::Swapchain::new(&device, &swapchain_loader, swapchain_info)?;

    let mut keyboard_state = KeyboardState::default();

    let mut camera = dolly::rig::CameraRig::builder()
        .with(dolly::drivers::Position::new(dolly::glam::Vec3::new(
            2.0, 4.0, 1.0,
        )))
        .with(dolly::drivers::YawPitch::new().pitch_degrees(-74.0))
        .with(dolly::drivers::Smooth::new_position_rotation(0.5, 0.25))
        .build();

    let mut cursor_grab = false;

    let mut screen_center =
        winit::dpi::LogicalPosition::new(extent.width as f64 / 2.0, extent.height as f64 / 2.0);

    let mut perspective_matrix = ultraviolet::projection::perspective_infinite_z_vk(
        59.0_f32.to_radians(),
        extent.width as f32 / extent.height as f32,
        0.1,
    );

    let command_buffer = init_command_buffer;
    let command_pool = init_command_pool;

    let semaphore_info = vk::SemaphoreCreateInfo::builder();
    let fence_info = vk::FenceCreateInfo::builder().flags(vk::FenceCreateFlags::SIGNALED);

    let present_semaphore = unsafe { device.create_semaphore(&semaphore_info, None)? };
    let render_semaphore = unsafe { device.create_semaphore(&semaphore_info, None)? };
    let render_fence = unsafe { device.create_fence(&fence_info, None)? };

    let mut draw_framebuffer = {
        let attachments = [hdr_framebuffer.view, depthbuffer.view];

        unsafe {
            device.create_framebuffer(
                &vk::FramebufferCreateInfo::builder()
                    .render_pass(render_passes.draw)
                    .attachments(&attachments)
                    .width(extent.width)
                    .height(extent.height)
                    .layers(1),
                None,
            )
        }
    }?;

    let mut tonemap_framebuffers =
        create_swapchain_image_framebuffers(&device, extent, &swapchain, &render_passes)?;

    let mut filter_constants = FilterConstants {
        threshold: 7.0,
        knee: 7.0,
    };

    event_loop.run(move |event, _, control_flow| {
        //egui_platform.handle_event(&event);

        let loop_closure = || -> anyhow::Result<()> {
            match event {
                Event::WindowEvent { event, .. } => match event {
                    WindowEvent::CloseRequested => *control_flow = ControlFlow::Exit,
                    WindowEvent::Resized(new_size) => {
                        extent.width = new_size.width;
                        extent.height = new_size.height;

                        perspective_matrix = ultraviolet::projection::perspective_infinite_z_vk(
                            59.0_f32.to_radians(),
                            extent.width as f32 / extent.height as f32,
                            0.1,
                        );

                        screen_center = winit::dpi::LogicalPosition::new(
                            extent.width as f64 / 2.0,
                            extent.height as f64 / 2.0,
                        );

                        swapchain_info.image_extent = extent;
                        swapchain_info.old_swapchain = swapchain.swapchain;

                        unsafe {
                            device.queue_wait_idle(queue)?;
                        }

                        swapchain = vulkan_common::Swapchain::new(
                            &device,
                            &swapchain_loader,
                            swapchain_info,
                        )?;

                        unsafe {
                            device.reset_command_pool(
                                command_pool,
                                vk::CommandPoolResetFlags::empty(),
                            )?;
                        }

                        unsafe {
                            device.begin_command_buffer(
                                command_buffer,
                                &vk::CommandBufferBeginInfo::builder()
                                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                            )?;
                        }

                        hdr_framebuffer.cleanup(&device, &mut allocator)?;
                        depthbuffer.cleanup(&device, &mut allocator)?;
                        bloom_texture.cleanup(&device, &mut allocator)?;

                        let mut init_resources = vulkan_common::InitResources {
                            command_buffer,
                            device: &device,
                            allocator: &mut allocator,
                            debug_utils_loader: Some(&debug_utils_loader),
                        };

                        hdr_framebuffer = create_hdr_framebuffer(
                            extent.width,
                            extent.height,
                            &mut init_resources,
                        )?;

                        depthbuffer =
                            create_depthbuffer(extent.width, extent.height, &mut init_resources)?;

                        let bloom_mips = bloom_mips_for_dimensions(extent.width, extent.height);

                        bloom_texture = BloomTextureWithMips::new_with_existing_sets(
                            &mut init_resources,
                            extent.width / 2,
                            extent.height / 2,
                            bloom_mips,
                            bloom_texture.storage_mips_descriptor_set,
                            bloom_texture.sampled_texture_descriptor_set,
                        )?;

                        drop(init_resources);

                        unsafe {
                            device.end_command_buffer(init_command_buffer)?;
                            let fence =
                                device.create_fence(&vk::FenceCreateInfo::builder(), None)?;

                            device.queue_submit(
                                queue,
                                &[*vk::SubmitInfo::builder()
                                    .command_buffers(&[init_command_buffer])],
                                fence,
                            )?;

                            device.wait_for_fences(&[fence], true, u64::MAX)?;

                            device.update_descriptor_sets(
                                &[
                                    *vk::WriteDescriptorSet::builder()
                                        .dst_set(tonemap_descriptor_set)
                                        .dst_binding(0)
                                        .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                                        .image_info(&[*vk::DescriptorImageInfo::builder()
                                            .image_view(hdr_framebuffer.view)
                                            .image_layout(
                                                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                                            )]),
                                    *vk::WriteDescriptorSet::builder()
                                        .dst_set(hdr_framebuffer_storage_ds)
                                        .dst_binding(0)
                                        .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                                        .image_info(&[*vk::DescriptorImageInfo::builder()
                                            .image_view(hdr_framebuffer.view)
                                            .image_layout(vk::ImageLayout::GENERAL)]),
                                ],
                                &[],
                            );
                        }

                        draw_framebuffer = {
                            let attachments = [hdr_framebuffer.view, depthbuffer.view];

                            unsafe {
                                device.create_framebuffer(
                                    &vk::FramebufferCreateInfo::builder()
                                        .render_pass(render_passes.draw)
                                        .attachments(&attachments)
                                        .width(extent.width)
                                        .height(extent.height)
                                        .layers(1),
                                    None,
                                )
                            }
                        }?;

                        tonemap_framebuffers = create_swapchain_image_framebuffers(&device, extent, &swapchain, &render_passes)?;
                    }
                    WindowEvent::KeyboardInput {
                        input:
                            KeyboardInput {
                                state,
                                virtual_keycode: Some(key),
                                ..
                            },
                        ..
                    } => {
                        let is_pressed = state == ElementState::Pressed;

                        match key {
                            VirtualKeyCode::W => keyboard_state.forwards = is_pressed,
                            VirtualKeyCode::S => keyboard_state.backwards = is_pressed,
                            VirtualKeyCode::A => keyboard_state.left = is_pressed,
                            VirtualKeyCode::D => keyboard_state.right = is_pressed,
                            VirtualKeyCode::F11 => {
                                if is_pressed {
                                    if window.fullscreen().is_some() {
                                        window.set_fullscreen(None);
                                    } else {
                                        window.set_fullscreen(Some(Fullscreen::Borderless(None)))
                                    }
                                }
                            }
                            VirtualKeyCode::G => {
                                if is_pressed {
                                    cursor_grab = !cursor_grab;

                                    if cursor_grab {
                                        window.set_cursor_position(screen_center)?;
                                    }

                                    window.set_cursor_visible(!cursor_grab);
                                    window.set_cursor_grab(cursor_grab)?;
                                }
                            }
                            _ => {}
                        }
                    }
                    WindowEvent::CursorMoved { position, .. } => {
                        if cursor_grab {
                            let position = position.to_logical::<f64>(window.scale_factor());

                            window.set_cursor_position(screen_center)?;

                            camera
                                .driver_mut::<dolly::drivers::YawPitch>()
                                .rotate_yaw_pitch(
                                    0.1 * (screen_center.x - position.x) as f32,
                                    0.1 * (screen_center.y - position.y) as f32,
                                );
                        }
                    }
                    _ => {}
                },
                Event::MainEventsCleared => {
                    let delta_time = 1.0 / 60.0;

                    let forwards = keyboard_state.forwards as i32 - keyboard_state.backwards as i32;
                    let right = keyboard_state.right as i32 - keyboard_state.left as i32;

                    let move_vec = camera.final_transform.rotation
                        * dolly::glam::Vec3::new(right as f32, 0.0, -forwards as f32)
                            .clamp_length_max(1.0);

                    camera
                        .driver_mut::<dolly::drivers::Position>()
                        .translate(move_vec * delta_time * 10.0);

                    camera.update(delta_time);

                    //egui_platform.update_time(delta_time as f64);

                    window.request_redraw();
                }
                Event::RedrawRequested(_) => {
                    unsafe {
                        device.wait_for_fences(&[render_fence], true, u64::MAX)?;

                        device.reset_fences(&[render_fence])?;

                        device
                            .reset_command_pool(command_pool, vk::CommandPoolResetFlags::empty())?;
                    }

                    match unsafe {
                        swapchain_loader.acquire_next_image(
                            swapchain.swapchain,
                            u64::MAX,
                            present_semaphore,
                            vk::Fence::null(),
                        )
                    } {
                        Ok((swapchain_image_index, _suboptimal)) => {
                            let draw_clear_values = [
                                vk::ClearValue {
                                    color: vk::ClearColorValue {
                                        float32: [0.0, 0.0, 0.0, 1.0],
                                    },
                                },
                                vk::ClearValue {
                                    depth_stencil: vk::ClearDepthStencilValue {
                                        depth: 1.0,
                                        stencil: 0,
                                    },
                                },
                            ];
                            let tonemap_clear_values = [vk::ClearValue {
                                color: vk::ClearColorValue {
                                    float32: [0.0, 0.0, 0.0, 1.0],
                                },
                            }];
                            let area = vk::Rect2D {
                                offset: vk::Offset2D { x: 0, y: 0 },
                                extent,
                            };

                            let draw_render_pass_info = vk::RenderPassBeginInfo::builder()
                                .render_pass(render_passes.draw)
                                .framebuffer(draw_framebuffer)
                                .render_area(area)
                                .clear_values(&draw_clear_values);

                            let tonemap_render_pass_info = vk::RenderPassBeginInfo::builder()
                                .render_pass(render_passes.tonemap)
                                .framebuffer(tonemap_framebuffers[swapchain_image_index as usize])
                                .render_area(area)
                                .clear_values(&tonemap_clear_values);

                            let viewport = *vk::Viewport::builder()
                                .x(0.0)
                                .y(0.0)
                                .width(extent.width as f32)
                                .height(extent.height as f32)
                                .min_depth(0.0)
                                .max_depth(1.0);

                            let camera_view = Mat4::look_at(
                                Vec3::from(<[f32; 3]>::from(camera.final_transform.position)),
                                Vec3::from(<[f32; 3]>::from(
                                    camera.final_transform.position
                                        + camera.final_transform.forward(),
                                )),
                                Vec3::from(<[f32; 3]>::from(camera.final_transform.up())),
                            );

                            unsafe {
                                device.begin_command_buffer(
                                    command_buffer,
                                    &vk::CommandBufferBeginInfo::builder()
                                        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                                )?;

                                device.cmd_begin_render_pass(
                                    command_buffer,
                                    &draw_render_pass_info,
                                    vk::SubpassContents::INLINE,
                                );

                                device.cmd_set_scissor(command_buffer, 0, &[area]);
                                device.cmd_set_viewport(command_buffer, 0, &[viewport]);

                                device.cmd_bind_pipeline(
                                    command_buffer,
                                    vk::PipelineBindPoint::GRAPHICS,
                                    graphics_pipelines.graphics_pipeline,
                                );

                                device.cmd_bind_descriptor_sets(
                                    command_buffer,
                                    vk::PipelineBindPoint::GRAPHICS,
                                    graphics_pipelines.general_pipeline_layout,
                                    0,
                                    &[descriptor_set],
                                    &[],
                                );
                                device.cmd_bind_vertex_buffers(
                                    command_buffer,
                                    0,
                                    &[vertex_buffer.buffer],
                                    &[0],
                                );
                                device.cmd_bind_index_buffer(
                                    command_buffer,
                                    index_buffer.buffer,
                                    0,
                                    vk::IndexType::UINT32,
                                );
                                device.cmd_push_constants(
                                    command_buffer,
                                    graphics_pipelines.general_pipeline_layout,
                                    vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                                    0,
                                    bytemuck::bytes_of(&(perspective_matrix * camera_view)),
                                );

                                device.cmd_draw_indexed(command_buffer, num_indices, 1, 0, 0, 0);

                                device.cmd_end_render_pass(command_buffer);

                                {
                                    prefilter_bloom(&device, command_buffer, extent, &bloom_texture, filter_constants, &compute_pipelines, tonemap_descriptor_set);

                                    let subresource_range = *vk::ImageSubresourceRange::builder()
                                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                                        .level_count(1)
                                        .layer_count(1);

                                    vk_sync::cmd::pipeline_barrier(
                                        &device,
                                        command_buffer,
                                        None,
                                        &[],
                                        &[vk_sync::ImageBarrier {
                                            previous_accesses: &[
                                                vk_sync::AccessType::ComputeShaderReadSampledImageOrUniformTexelBuffer,
                                            ],
                                            next_accesses: &[vk_sync::AccessType::ComputeShaderWrite],
                                            next_layout: vk_sync::ImageLayout::General,
                                            image: hdr_framebuffer.image,
                                            range: subresource_range,
                                            ..Default::default()
                                        }],
                                    );

                                    compute_bloom(
                                        &device,
                                        command_buffer,
                                        extent,
                                        &bloom_texture,
                                        &compute_pipelines,
                                        hdr_framebuffer_storage_ds,
                                    );

                                    vk_sync::cmd::pipeline_barrier(
                                        &device,
                                        command_buffer,
                                        None,
                                        &[],
                                        &[vk_sync::ImageBarrier {
                                            previous_accesses: &[vk_sync::AccessType::ComputeShaderWrite],
                                            next_accesses: &[
                                                vk_sync::AccessType::FragmentShaderReadSampledImageOrUniformTexelBuffer,
                                            ],
                                            next_layout: vk_sync::ImageLayout::Optimal,
                                            image: hdr_framebuffer.image,
                                            range: subresource_range,
                                            ..Default::default()
                                        }],
                                    );
                                }

                                device.cmd_begin_render_pass(
                                    command_buffer,
                                    &tonemap_render_pass_info,
                                    vk::SubpassContents::INLINE,
                                );

                                device.cmd_bind_pipeline(
                                    command_buffer,
                                    vk::PipelineBindPoint::GRAPHICS,
                                    graphics_pipelines.tonemap_pipeline,
                                );

                                device.cmd_bind_descriptor_sets(
                                    command_buffer,
                                    vk::PipelineBindPoint::GRAPHICS,
                                    graphics_pipelines.general_pipeline_layout,
                                    0,
                                    &[tonemap_descriptor_set],
                                    &[],
                                );

                                device.cmd_push_constants(
                                    command_buffer,
                                    graphics_pipelines.general_pipeline_layout,
                                    vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                                    0,
                                    bytemuck::bytes_of(
                                        &colstodian::tonemap::BakedLottesTonemapperParams::from(
                                            colstodian::tonemap::LottesTonemapperParams {
                                                ..Default::default()
                                            },
                                        ),
                                    ),
                                );

                                device.cmd_draw(command_buffer, 3, 1, 0, 0);

                                device.cmd_end_render_pass(command_buffer);

                                device.end_command_buffer(command_buffer)?;

                                device.queue_submit(
                                    queue,
                                    &[*vk::SubmitInfo::builder()
                                        .wait_semaphores(&[present_semaphore])
                                        .wait_dst_stage_mask(&[vk::PipelineStageFlags::TRANSFER])
                                        .command_buffers(&[command_buffer])
                                        .signal_semaphores(&[render_semaphore])],
                                    render_fence,
                                )?;

                                swapchain_loader.queue_present(
                                    queue,
                                    &vk::PresentInfoKHR::builder()
                                        .wait_semaphores(&[render_semaphore])
                                        .swapchains(&[swapchain.swapchain])
                                        .image_indices(&[swapchain_image_index]),
                                )?;
                            }
                        }
                        Err(error) => log::warn!("Next frame error: {:?}", error),
                    }
                }
                Event::LoopDestroyed => {
                    unsafe {
                        device.queue_wait_idle(queue)?;
                    }

                    texture.cleanup(&device, &mut allocator)?;
                    hdr_framebuffer.cleanup(&device, &mut allocator)?;
                    depthbuffer.cleanup(&device, &mut allocator)?;
                    vertex_buffer.cleanup(&device, &mut allocator)?;
                    index_buffer.cleanup(&device, &mut allocator)?;
                    bloom_texture.cleanup(&device, &mut allocator)?;
                }
                _ => {}
            }

            Ok(())
        };

        if let Err(loop_closure) = loop_closure() {
            log::error!("Error: {}", loop_closure);
        }
    });
}

unsafe fn prefilter_bloom(
    device: &ash::Device,
    command_buffer: vk::CommandBuffer,
    extent: vk::Extent2D,
    bloom_texture: &BloomTextureWithMips,
    filter_constants: FilterConstants,
    compute_pipelines: &ComputePipelines,
    input_descriptor_set: vk::DescriptorSet,
) {
    device.cmd_bind_pipeline(
        command_buffer,
        vk::PipelineBindPoint::COMPUTE,
        compute_pipelines.downsample_initial,
    );

    device.cmd_bind_descriptor_sets(
        command_buffer,
        vk::PipelineBindPoint::COMPUTE,
        compute_pipelines.bloom_pipeline_layout,
        0,
        &[
            input_descriptor_set,
            bloom_texture.storage_mips_descriptor_set,
        ],
        &[],
    );

    device.cmd_push_constants(
        command_buffer,
        compute_pipelines.bloom_pipeline_layout,
        vk::ShaderStageFlags::COMPUTE,
        0,
        bytemuck::bytes_of(&filter_constants),
    );

    device.cmd_dispatch(
        command_buffer,
        dispatch_count(extent.width >> 1, 8),
        dispatch_count(extent.height >> 1, 8),
        1,
    );
}

unsafe fn compute_bloom(
    device: &ash::Device,
    command_buffer: vk::CommandBuffer,
    extent: vk::Extent2D,
    bloom_texture: &BloomTextureWithMips,
    compute_pipelines: &ComputePipelines,
    output_descriptor_set: vk::DescriptorSet,
) {
    let bloom_mips = bloom_mips_for_dimensions(extent.width, extent.height);

    device.cmd_bind_pipeline(
        command_buffer,
        vk::PipelineBindPoint::COMPUTE,
        compute_pipelines.downsample,
    );

    device.cmd_bind_descriptor_sets(
        command_buffer,
        vk::PipelineBindPoint::COMPUTE,
        compute_pipelines.bloom_pipeline_layout,
        0,
        &[bloom_texture.sampled_texture_descriptor_set],
        &[],
    );

    for i in 0..bloom_mips - 1 {
        device.cmd_push_constants(
            command_buffer,
            compute_pipelines.bloom_pipeline_layout,
            vk::ShaderStageFlags::COMPUTE,
            0,
            bytemuck::bytes_of(&(i as u32)),
        );
        device.cmd_dispatch(
            command_buffer,
            dispatch_count(extent.width >> (i + 2), 8),
            dispatch_count(extent.height >> (i + 2), 8),
            1,
        );

        let subresource_range = *vk::ImageSubresourceRange::builder()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .level_count(1)
            .layer_count(1)
            .base_mip_level(i + 1);

        vk_sync::cmd::pipeline_barrier(
            &device,
            command_buffer,
            None,
            &[],
            &[vk_sync::ImageBarrier {
                previous_accesses: &[vk_sync::AccessType::ComputeShaderWrite],
                next_accesses: &[
                    vk_sync::AccessType::ComputeShaderReadSampledImageOrUniformTexelBuffer,
                ],
                next_layout: vk_sync::ImageLayout::General,
                image: bloom_texture.image.image,
                range: subresource_range,
                ..Default::default()
            }],
        );
    }

    device.cmd_bind_pipeline(
        command_buffer,
        vk::PipelineBindPoint::COMPUTE,
        compute_pipelines.upsample,
    );

    for i in (0..bloom_mips - 1).rev() {
        device.cmd_push_constants(
            command_buffer,
            compute_pipelines.bloom_pipeline_layout,
            vk::ShaderStageFlags::COMPUTE,
            0,
            bytemuck::bytes_of(&(i as u32)),
        );
        device.cmd_dispatch(
            command_buffer,
            dispatch_count(extent.width >> (i + 1), 8),
            dispatch_count(extent.height >> (i + 1), 8),
            1,
        );

        let subresource_range = *vk::ImageSubresourceRange::builder()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .level_count(1)
            .layer_count(1)
            .base_mip_level(i);

        vk_sync::cmd::pipeline_barrier(
            &device,
            command_buffer,
            None,
            &[],
            &[vk_sync::ImageBarrier {
                previous_accesses: &[vk_sync::AccessType::ComputeShaderWrite],
                next_accesses: &[
                    vk_sync::AccessType::ComputeShaderReadSampledImageOrUniformTexelBuffer,
                ],
                next_layout: vk_sync::ImageLayout::General,
                image: bloom_texture.image.image,
                range: subresource_range,
                ..Default::default()
            }],
        );
    }

    device.cmd_bind_pipeline(
        command_buffer,
        vk::PipelineBindPoint::COMPUTE,
        compute_pipelines.upsample_final,
    );

    device.cmd_bind_descriptor_sets(
        command_buffer,
        vk::PipelineBindPoint::COMPUTE,
        compute_pipelines.upsample_final_pipeline_layout,
        0,
        &[
            bloom_texture.sampled_texture_descriptor_set,
            output_descriptor_set,
        ],
        &[],
    );

    device.cmd_dispatch(
        command_buffer,
        dispatch_count(extent.width, 8),
        dispatch_count(extent.height, 8),
        1,
    );
}

fn create_swapchain_image_framebuffers(
    device: &ash::Device,
    extent: vk::Extent2D,
    swapchain: &vulkan_common::Swapchain,
    render_passes: &RenderPasses,
) -> anyhow::Result<Vec<vk::Framebuffer>> {
    swapchain
        .image_views
        .iter()
        .map(|image_view| {
            unsafe {
                device.create_framebuffer(
                    &vk::FramebufferCreateInfo::builder()
                        .render_pass(render_passes.tonemap)
                        .attachments(&[*image_view])
                        .width(extent.width)
                        .height(extent.height)
                        .layers(1),
                    None,
                )
            }
            .map_err(|err| err.into())
        })
        .collect()
}

fn create_hdr_framebuffer(
    width: u32,
    height: u32,
    init_resources: &mut vulkan_common::InitResources,
) -> anyhow::Result<vulkan_common::Image> {
    vulkan_common::Image::new(
        &vulkan_common::ImageDescriptor {
            width,
            height,
            name: "hdr framebuffer",
            mip_levels: 1,
            format: vk::Format::R16G16B16A16_SFLOAT,
            usage: vk::ImageUsageFlags::COLOR_ATTACHMENT
                | vk::ImageUsageFlags::SAMPLED
                | vk::ImageUsageFlags::STORAGE,
            next_accesses: &[vk_sync::AccessType::ColorAttachmentWrite],
            next_layout: vk_sync::ImageLayout::Optimal,
        },
        init_resources,
    )
}

fn create_depthbuffer(
    width: u32,
    height: u32,
    init_resources: &mut vulkan_common::InitResources,
) -> anyhow::Result<vulkan_common::Image> {
    vulkan_common::Image::new(
        &vulkan_common::ImageDescriptor {
            width,
            height,
            name: "depth",
            mip_levels: 1,
            format: vk::Format::D32_SFLOAT,
            usage: vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
            next_accesses: &[vk_sync::AccessType::DepthStencilAttachmentWrite],
            next_layout: vk_sync::ImageLayout::Optimal,
        },
        init_resources,
    )
}

struct BloomTextureWithMips {
    image: vulkan_common::Image,
    views: Vec<vk::ImageView>,
    storage_mips_descriptor_set: vk::DescriptorSet,
    sampled_texture_descriptor_set: vk::DescriptorSet,
}

impl BloomTextureWithMips {
    fn new(
        init_resources: &mut vulkan_common::InitResources,
        width: u32,
        height: u32,
        mip_levels: u32,
        descriptor_set_layouts: &DescriptorSetLayouts,
        descriptor_pool: vk::DescriptorPool,
        sampler: vk::Sampler,
    ) -> anyhow::Result<Self> {
        let descriptor_sets = unsafe {
            init_resources.device.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::builder()
                    .set_layouts(&[
                        descriptor_set_layouts.bloom_texture,
                        descriptor_set_layouts.sampled_texture,
                    ])
                    .descriptor_pool(descriptor_pool),
            )
        }?;

        let storage_mips_descriptor_set = descriptor_sets[0];
        let sampled_texture_descriptor_set = descriptor_sets[1];

        unsafe {
            init_resources.device.update_descriptor_sets(
                &[*vk::WriteDescriptorSet::builder()
                    .dst_set(sampled_texture_descriptor_set)
                    .dst_binding(1)
                    .descriptor_type(vk::DescriptorType::SAMPLER)
                    .image_info(&[*vk::DescriptorImageInfo::builder().sampler(sampler)])],
                &[],
            );
        }

        Self::new_with_existing_sets(
            init_resources,
            width,
            height,
            mip_levels,
            storage_mips_descriptor_set,
            sampled_texture_descriptor_set,
        )
    }

    fn new_with_existing_sets(
        init_resources: &mut vulkan_common::InitResources,
        width: u32,
        height: u32,
        mip_levels: u32,
        storage_mips_descriptor_set: vk::DescriptorSet,
        sampled_texture_descriptor_set: vk::DescriptorSet,
    ) -> anyhow::Result<Self> {
        let image = vulkan_common::Image::new(
            &vulkan_common::ImageDescriptor {
                width,
                height,
                name: "bloom texture",
                mip_levels,
                format: vk::Format::R16G16B16A16_SFLOAT,
                usage: vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::SAMPLED,
                next_accesses: &[vk_sync::AccessType::ComputeShaderWrite],
                next_layout: vk_sync::ImageLayout::Optimal,
            },
            init_resources,
        )?;

        let device = init_resources.device;

        let views = (0..mip_levels)
            .map(|i| {
                let subresource_range = *vk::ImageSubresourceRange::builder()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .level_count(1)
                    .layer_count(1)
                    .base_mip_level(i);

                unsafe {
                    device.create_image_view(
                        &vk::ImageViewCreateInfo::builder()
                            .image(image.image)
                            .view_type(vk::ImageViewType::TYPE_2D)
                            .format(vk::Format::R16G16B16A16_SFLOAT)
                            .subresource_range(subresource_range),
                        None,
                    )
                }
            })
            .collect::<Result<Vec<_>, _>>()?;

        let dummy_mips = std::iter::repeat(mip_levels - 1).take((MAX_MIPS - mip_levels) as usize);

        let image_infos: Vec<_> = (0..mip_levels)
            .chain(dummy_mips)
            .map(|i| {
                *vk::DescriptorImageInfo::builder()
                    .image_view(views[i as usize])
                    .image_layout(vk::ImageLayout::GENERAL)
            })
            .collect();

        unsafe {
            device.update_descriptor_sets(
                &[
                    *vk::WriteDescriptorSet::builder()
                        .dst_set(storage_mips_descriptor_set)
                        .dst_binding(0)
                        .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                        .image_info(&image_infos),
                    *vk::WriteDescriptorSet::builder()
                        .dst_set(sampled_texture_descriptor_set)
                        .dst_binding(0)
                        .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                        .image_info(&[*vk::DescriptorImageInfo::builder()
                            .image_view(image.view)
                            .image_layout(vk::ImageLayout::GENERAL)]),
                ],
                &[],
            );
        }

        Ok(Self {
            image,
            views,
            storage_mips_descriptor_set,
            sampled_texture_descriptor_set,
        })
    }

    fn cleanup(
        &self,
        device: &ash::Device,
        allocator: &mut gpu_allocator::vulkan::Allocator,
    ) -> anyhow::Result<()> {
        for view in &self.views {
            unsafe {
                device.destroy_image_view(*view, None);
            }
        }

        self.image.cleanup(device, allocator)
    }
}

#[derive(Default)]
struct KeyboardState {
    forwards: bool,
    right: bool,
    backwards: bool,
    left: bool,
}

struct RenderPasses {
    draw: vk::RenderPass,
    tonemap: vk::RenderPass,
}

impl RenderPasses {
    fn new(device: &ash::Device, surface_format: vk::Format) -> anyhow::Result<Self> {
        let draw_attachments = [
            // HDR framebuffer
            *vk::AttachmentDescription::builder()
                .format(vk::Format::R16G16B16A16_SFLOAT)
                .samples(vk::SampleCountFlags::TYPE_1)
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::STORE)
                .final_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL),
            // Depth buffer
            *vk::AttachmentDescription::builder()
                .format(vk::Format::D32_SFLOAT)
                .samples(vk::SampleCountFlags::TYPE_1)
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::STORE)
                .final_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL),
        ];

        let hdr_framebuffer_ref = [*vk::AttachmentReference::builder()
            .attachment(0)
            .layout(vk::ImageLayout::GENERAL)];

        let depth_attachment_ref = *vk::AttachmentReference::builder()
            .attachment(1)
            .layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);

        let draw_subpass = [*vk::SubpassDescription::builder()
            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
            .color_attachments(&hdr_framebuffer_ref)
            .depth_stencil_attachment(&depth_attachment_ref)];

        let draw_subpass_dependency = [*vk::SubpassDependency::builder()
            .src_subpass(vk::SUBPASS_EXTERNAL)
            .dst_subpass(0)
            .src_stage_mask(vk::PipelineStageFlags::TOP_OF_PIPE)
            .src_access_mask(vk::AccessFlags::empty())
            .dst_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
            .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)];

        let draw_render_pass = unsafe {
            device.create_render_pass(
                &vk::RenderPassCreateInfo::builder()
                    .attachments(&draw_attachments)
                    .subpasses(&draw_subpass)
                    .dependencies(&draw_subpass_dependency),
                None,
            )
        }?;

        let tonemap_attachments = [
            // Swapchain framebuffer
            *vk::AttachmentDescription::builder()
                .format(surface_format)
                .samples(vk::SampleCountFlags::TYPE_1)
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::STORE)
                .final_layout(vk::ImageLayout::PRESENT_SRC_KHR),
        ];

        let swapchain_framebuffer_ref = [*vk::AttachmentReference::builder()
            .attachment(0)
            .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)];

        let tonemap_subpass = [*vk::SubpassDescription::builder()
            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
            .color_attachments(&swapchain_framebuffer_ref)];

        let tonemap_subpass_dependency = [*vk::SubpassDependency::builder()
            .src_subpass(vk::SUBPASS_EXTERNAL)
            .dst_subpass(0)
            .src_stage_mask(vk::PipelineStageFlags::TOP_OF_PIPE)
            .src_access_mask(vk::AccessFlags::empty())
            .dst_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
            .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)];

        let tonemap_render_pass = unsafe {
            device.create_render_pass(
                &vk::RenderPassCreateInfo::builder()
                    .attachments(&tonemap_attachments)
                    .subpasses(&tonemap_subpass)
                    .dependencies(&tonemap_subpass_dependency),
                None,
            )
        }?;

        Ok(Self {
            draw: draw_render_pass,
            tonemap: tonemap_render_pass,
        })
    }
}

struct GraphicsPipelines {
    graphics_pipeline: vk::Pipeline,
    tonemap_pipeline: vk::Pipeline,
    general_pipeline_layout: vk::PipelineLayout,
}

impl GraphicsPipelines {
    fn new(
        device: &ash::Device,
        descriptor_set_layouts: &DescriptorSetLayouts,
        render_passes: &RenderPasses,
        pipeline_cache: vk::PipelineCache,
    ) -> anyhow::Result<Self> {
        use vulkan_common::load_shader_module_as_stage;

        let vertex_entry_point = CString::new("vertex")?;

        let vertex_stage = load_shader_module_as_stage(
            include_bytes!("../../shaders/vertex.spv"),
            vk::ShaderStageFlags::VERTEX,
            device,
            &vertex_entry_point,
        )?;

        let fragment_entry_point = CString::new("fragment")?;

        let fragment_stage = load_shader_module_as_stage(
            include_bytes!("../../shaders/fragment.spv"),
            vk::ShaderStageFlags::FRAGMENT,
            device,
            &fragment_entry_point,
        )?;

        let fullscreen_tri_entry_point = CString::new("fullscreen_tri")?;

        let fullscreen_tri_stage = load_shader_module_as_stage(
            include_bytes!("../../shaders/fullscreen_tri.spv"),
            vk::ShaderStageFlags::VERTEX,
            device,
            &fullscreen_tri_entry_point,
        )?;

        let tonemap_entry_point = CString::new("tonemap")?;

        let tonemap_stage = load_shader_module_as_stage(
            include_bytes!("../../shaders/tonemap.spv"),
            vk::ShaderStageFlags::FRAGMENT,
            device,
            &tonemap_entry_point,
        )?;

        let general_pipeline_layout = unsafe {
            device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::builder()
                    .set_layouts(&[descriptor_set_layouts.sampled_texture])
                    .push_constant_ranges(&[*vk::PushConstantRange::builder()
                        .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT)
                        .size(128)]),
                None,
            )
        }?;

        let pipelines = {
            let stages = &[*vertex_stage, *fragment_stage];

            let graphics_pipeline_desc = vulkan_common::GraphicsPipelineDescriptor {
                primitive_state: vulkan_common::PrimitiveState {
                    cull_mode: vk::CullModeFlags::BACK,
                },
                depth_stencil_state: vulkan_common::DepthStencilState {
                    depth_test_enable: true,
                    depth_write_enable: true,
                    depth_compare_op: vk::CompareOp::LESS,
                },
                vertex_attributes: &vulkan_common::create_vertex_attribute_descriptions(
                    0,
                    &[
                        vulkan_common::VertexAttribute::Vec3,
                        vulkan_common::VertexAttribute::Vec2,
                    ],
                ),
                vertex_bindings: &[*vk::VertexInputBindingDescription::builder()
                    .binding(0)
                    .stride(std::mem::size_of::<Vertex>() as u32)],
                colour_attachments: &[*vk::PipelineColorBlendAttachmentState::builder()
                    .color_write_mask(vk::ColorComponentFlags::all())
                    .blend_enable(false)],
            };

            let baked = graphics_pipeline_desc.as_baked();

            let graphics_pipeline_desc = baked.as_pipeline_create_info(
                stages,
                general_pipeline_layout,
                render_passes.draw,
                0,
            );

            let tonemap_pipeline = vulkan_common::GraphicsPipelineDescriptor {
                primitive_state: vulkan_common::PrimitiveState {
                    cull_mode: vk::CullModeFlags::NONE,
                },
                depth_stencil_state: vulkan_common::DepthStencilState {
                    depth_test_enable: false,
                    depth_write_enable: false,
                    depth_compare_op: vk::CompareOp::ALWAYS,
                },
                vertex_attributes: &[],
                vertex_bindings: &[],
                colour_attachments: &[*vk::PipelineColorBlendAttachmentState::builder()
                    .color_write_mask(vk::ColorComponentFlags::all())
                    .blend_enable(false)],
            };

            let tonemap_baked = tonemap_pipeline.as_baked();
            let tonemap_stages = &[*fullscreen_tri_stage, *tonemap_stage];

            let tonemap_desc = tonemap_baked.as_pipeline_create_info(
                tonemap_stages,
                general_pipeline_layout,
                render_passes.tonemap,
                0,
            );

            unsafe {
                device.create_graphics_pipelines(
                    pipeline_cache,
                    &[*graphics_pipeline_desc, *tonemap_desc],
                    None,
                )
            }
            .map_err(|(_, err)| err)?
        };

        Ok(Self {
            graphics_pipeline: pipelines[0],
            tonemap_pipeline: pipelines[1],
            general_pipeline_layout,
        })
    }
}

struct ComputePipelines {
    downsample_initial: vk::Pipeline,
    downsample: vk::Pipeline,
    upsample: vk::Pipeline,
    upsample_final: vk::Pipeline,
    upsample_final_pipeline_layout: vk::PipelineLayout,
    bloom_pipeline_layout: vk::PipelineLayout,
}

impl ComputePipelines {
    fn new(
        device: &ash::Device,
        descriptor_set_layouts: &DescriptorSetLayouts,
        pipeline_cache: vk::PipelineCache,
    ) -> anyhow::Result<Self> {
        let bloom_pipeline_layout = unsafe {
            device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::builder()
                    .set_layouts(&[
                        descriptor_set_layouts.sampled_texture,
                        descriptor_set_layouts.bloom_texture,
                    ])
                    .push_constant_ranges(&[*vk::PushConstantRange::builder()
                        .stage_flags(vk::ShaderStageFlags::COMPUTE)
                        .size(128)]),
                None,
            )
        }?;

        let upsample_final_pipeline_layout = unsafe {
            device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::builder()
                    .set_layouts(&[
                        descriptor_set_layouts.sampled_texture,
                        descriptor_set_layouts.storage_texture,
                    ])
                    .push_constant_ranges(&[]),
                None,
            )
        }?;

        let downsample_initial_entry_point = CString::new("downsample_initial")?;

        let downsample_initial_stage = vulkan_common::load_shader_module_as_stage(
            include_bytes!("../../shaders/downsample_initial.spv"),
            vk::ShaderStageFlags::COMPUTE,
            device,
            &downsample_initial_entry_point,
        )?;

        let downsample_initial_create_info = vk::ComputePipelineCreateInfo::builder()
            .stage(*downsample_initial_stage)
            .layout(bloom_pipeline_layout);

        let downsample_entry_point = CString::new("downsample")?;

        let downsample_stage = vulkan_common::load_shader_module_as_stage(
            include_bytes!("../../shaders/downsample.spv"),
            vk::ShaderStageFlags::COMPUTE,
            device,
            &downsample_entry_point,
        )?;

        let downsample_create_info = vk::ComputePipelineCreateInfo::builder()
            .stage(*downsample_stage)
            .layout(bloom_pipeline_layout);

        let upsample_entry_point = CString::new("upsample")?;

        let upsample_stage = vulkan_common::load_shader_module_as_stage(
            include_bytes!("../../shaders/upsample.spv"),
            vk::ShaderStageFlags::COMPUTE,
            device,
            &upsample_entry_point,
        )?;

        let upsample_create_info = vk::ComputePipelineCreateInfo::builder()
            .stage(*upsample_stage)
            .layout(bloom_pipeline_layout);

        let upsample_final_entry_point = CString::new("upsample_final")?;

        let upsample_final_stage = vulkan_common::load_shader_module_as_stage(
            include_bytes!("../../shaders/upsample_final.spv"),
            vk::ShaderStageFlags::COMPUTE,
            device,
            &upsample_final_entry_point,
        )?;

        let upsample_final_create_info = vk::ComputePipelineCreateInfo::builder()
            .stage(*upsample_final_stage)
            .layout(upsample_final_pipeline_layout);

        let pipelines = unsafe {
            device.create_compute_pipelines(
                pipeline_cache,
                &[
                    *downsample_initial_create_info,
                    *downsample_create_info,
                    *upsample_create_info,
                    *upsample_final_create_info,
                ],
                None,
            )
        }
        .map_err(|(_, err)| err)?;

        Ok(Self {
            bloom_pipeline_layout,
            downsample_initial: pipelines[0],
            downsample: pipelines[1],
            upsample: pipelines[2],
            upsample_final: pipelines[3],
            upsample_final_pipeline_layout,
        })
    }
}

struct DescriptorSetLayouts {
    sampled_texture: vk::DescriptorSetLayout,
    bloom_texture: vk::DescriptorSetLayout,
    storage_texture: vk::DescriptorSetLayout,
}

impl DescriptorSetLayouts {
    fn new(device: &ash::Device) -> anyhow::Result<Self> {
        unsafe {
            Ok(Self {
                sampled_texture: device.create_descriptor_set_layout(
                    &*vk::DescriptorSetLayoutCreateInfo::builder().bindings(&[
                        *vk::DescriptorSetLayoutBinding::builder()
                            .binding(0)
                            .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                            .descriptor_count(1)
                            .stage_flags(
                                vk::ShaderStageFlags::FRAGMENT | vk::ShaderStageFlags::COMPUTE,
                            ),
                        *vk::DescriptorSetLayoutBinding::builder()
                            .binding(1)
                            .descriptor_type(vk::DescriptorType::SAMPLER)
                            .descriptor_count(1)
                            .stage_flags(
                                vk::ShaderStageFlags::FRAGMENT | vk::ShaderStageFlags::COMPUTE,
                            ),
                    ]),
                    None,
                )?,
                bloom_texture: device.create_descriptor_set_layout(
                    &*vk::DescriptorSetLayoutCreateInfo::builder().bindings(&[
                        *vk::DescriptorSetLayoutBinding::builder()
                            .binding(0)
                            .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                            .descriptor_count(MAX_MIPS)
                            .stage_flags(vk::ShaderStageFlags::COMPUTE),
                    ]),
                    None,
                )?,
                storage_texture: device.create_descriptor_set_layout(
                    &*vk::DescriptorSetLayoutCreateInfo::builder().bindings(&[
                        *vk::DescriptorSetLayoutBinding::builder()
                            .binding(0)
                            .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                            .descriptor_count(1)
                            .stage_flags(vk::ShaderStageFlags::COMPUTE),
                    ]),
                    None,
                )?,
            })
        }
    }
}

#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
struct Vertex {
    position: Vec3,
    uv: Vec2,
}

fn load_gltf_from_bytes(
    bytes: &[u8],
    init_resources: &mut vulkan_common::InitResources,
    buffers_to_cleanup: &mut Vec<vulkan_common::Buffer>,
) -> anyhow::Result<(
    vulkan_common::Buffer,
    vulkan_common::Buffer,
    u32,
    vulkan_common::Image,
)> {
    let gltf = gltf::Gltf::from_slice(bytes)?;

    let buffer_blob = gltf.blob.as_ref().unwrap();

    let mut indices = Vec::new();
    let mut vertices = Vec::new();

    for mesh in gltf.meshes() {
        for primitive in mesh.primitives() {
            let reader = primitive.reader(|_| Some(buffer_blob));

            let read_indices = reader.read_indices().unwrap().into_u32();

            let num_vertices = vertices.len() as u32;

            indices.extend(read_indices.map(|index| index + num_vertices));

            let positions = reader.read_positions().unwrap();
            let uvs = reader.read_tex_coords(0).unwrap().into_f32();

            for (position, uv) in positions.zip(uvs) {
                vertices.push(Vertex {
                    position: position.into(),
                    uv: uv.into(),
                });
            }
        }
    }

    let material = gltf.materials().next().unwrap();

    let texture = material.emissive_texture().unwrap();

    let texture = load_texture_from_gltf(
        texture.texture(),
        "emissive texture",
        buffer_blob,
        init_resources,
        buffers_to_cleanup,
    )?;

    let num_indices = indices.len() as u32;

    let vertices = vulkan_common::Buffer::new(
        bytemuck::cast_slice(&vertices),
        "vertices",
        vk::BufferUsageFlags::VERTEX_BUFFER,
        init_resources,
    )?;

    let indices = vulkan_common::Buffer::new(
        bytemuck::cast_slice(&indices),
        "indices",
        vk::BufferUsageFlags::INDEX_BUFFER,
        init_resources,
    )?;

    Ok((vertices, indices, num_indices, texture))
}

fn load_texture_from_gltf(
    texture: gltf::texture::Texture,
    label: &str,
    buffer_blob: &[u8],
    init_resources: &mut vulkan_common::InitResources,
    buffers_to_cleanup: &mut Vec<vulkan_common::Buffer>,
) -> anyhow::Result<vulkan_common::Image> {
    let texture_view = match texture.source().source() {
        gltf::image::Source::View { view, .. } => view,
        _ => {
            return Err(anyhow::anyhow!(
                "Image source is a uri which we don't support"
            ))
        }
    };

    let texture_start = texture_view.offset();
    let texture_end = texture_start + texture_view.length();
    let texture_bytes = &buffer_blob[texture_start..texture_end];

    let decoded_bytes =
        image::load_from_memory_with_format(texture_bytes, image::ImageFormat::Png)?;

    let decoded_rgba8 = decoded_bytes.to_rgba8();

    let (image, staging_buffer) = vulkan_common::create_image_from_bytes(
        &*decoded_rgba8,
        vk::Extent3D {
            width: decoded_rgba8.width(),
            height: decoded_rgba8.height(),
            depth: 1,
        },
        vk::ImageViewType::TYPE_2D,
        vk::Format::R8G8B8A8_SRGB,
        label,
        init_resources,
        &[vk_sync::AccessType::FragmentShaderReadSampledImageOrUniformTexelBuffer],
        vk_sync::ImageLayout::Optimal,
    )?;

    buffers_to_cleanup.push(staging_buffer);

    Ok(image)
}

fn bloom_mips_for_dimensions(width: u32, height: u32) -> u32 {
    let mut mips = 1;

    while (width.min(height) >> (mips + 1)) > 0 {
        mips += 1;
    }

    mips
}

const fn dispatch_count(num: u32, group_size: u32) -> u32 {
    let mut count = num / group_size;
    let rem = num % group_size;
    if rem != 0 {
        count += 1;
    }

    count
}
