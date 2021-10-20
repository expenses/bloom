use ultraviolet::{Mat4, Vec2, Vec3};
use wgpu::util::DeviceExt;
use winit::event::{ElementState, Event, KeyboardInput, VirtualKeyCode, WindowEvent};
use winit::event_loop::ControlFlow;
use winit::window::Fullscreen;

fn main() -> anyhow::Result<()> {
    let instance = wgpu::Instance::new(wgpu::Backends::PRIMARY);

    let event_loop = winit::event_loop::EventLoop::new();
    let window = winit::window::Window::new(&event_loop)?;

    let surface = unsafe { instance.create_surface(&window) };

    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: Some(&surface),
        ..Default::default()
    }))
    .ok_or_else(|| anyhow::anyhow!("Failed to get adapter"))?;

    let surface_format = surface
        .get_preferred_format(&adapter)
        .expect("unreachable; adapter was chosen to be compatible with the surface");

    let (device, queue) = pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            label: Some("device"),
            features: wgpu::Features::PUSH_CONSTANTS
                | wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES
                | wgpu::Features::TEXTURE_BINDING_ARRAY
                | wgpu::Features::STORAGE_RESOURCE_BINDING_ARRAY
                | wgpu::Features::SPIRV_SHADER_PASSTHROUGH,
            limits: wgpu::Limits {
                max_push_constant_size: 128,
                max_storage_textures_per_shader_stage: 16,
                ..Default::default()
            },
        },
        None,
    ))?;

    let window_size = window.inner_size();

    let mut surface_configuration = wgpu::SurfaceConfiguration {
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        format: surface_format,
        width: window_size.width,
        height: window_size.height,
        present_mode: wgpu::PresentMode::Fifo,
    };

    surface.configure(&device, &surface_configuration);

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("bind group layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT | wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::FRAGMENT | wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Sampler {
                    filtering: true,
                    comparison: false,
                },
                count: None,
            },
        ],
    });

    let base_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("base pipeline layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[wgpu::PushConstantRange {
            stages: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
            range: 0..128,
        }],
    });

    let vertex_module = unsafe {
        device.create_shader_module_spirv(&wgpu::include_spirv_raw!("../shaders/vertex.spv"))
    };
    let fragment_module = unsafe {
        device.create_shader_module_spirv(&wgpu::include_spirv_raw!("../shaders/fragment.spv"))
    };

    let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("render pipeline"),
        layout: Some(&base_pipeline_layout),
        vertex: wgpu::VertexState {
            module: &vertex_module,
            entry_point: "vertex",
            buffers: &[wgpu::VertexBufferLayout {
                array_stride: std::mem::size_of::<Vertex>() as u64,
                step_mode: wgpu::VertexStepMode::Vertex,
                attributes: &wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x2],
            }],
        },
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            cull_mode: Some(wgpu::Face::Back),
            ..Default::default()
        },
        fragment: Some(wgpu::FragmentState {
            module: &fragment_module,
            entry_point: "fragment",
            targets: &[wgpu::ColorTargetState {
                format: wgpu::TextureFormat::Rgba32Float,
                write_mask: wgpu::ColorWrites::COLOR,
                blend: None,
            }],
        }),
        depth_stencil: Some(wgpu::DepthStencilState {
            format: wgpu::TextureFormat::Depth32Float,
            depth_write_enabled: true,
            depth_compare: wgpu::CompareFunction::Less,
            stencil: Default::default(),
            bias: Default::default(),
        }),
        multisample: wgpu::MultisampleState::default(),
    });

    let fullscreen_tri_module = unsafe {
        device
            .create_shader_module_spirv(&wgpu::include_spirv_raw!("../shaders/fullscreen_tri.spv"))
    };
    let tonemap_module = unsafe {
        device.create_shader_module_spirv(&wgpu::include_spirv_raw!("../shaders/tonemap.spv"))
    };

    let tonemap_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("tonemap pipeline"),
        layout: Some(&base_pipeline_layout),
        vertex: wgpu::VertexState {
            module: &fullscreen_tri_module,
            entry_point: "fullscreen_tri",
            buffers: &[],
        },
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            ..Default::default()
        },
        fragment: Some(wgpu::FragmentState {
            module: &tonemap_module,
            entry_point: "tonemap",
            targets: &[wgpu::ColorTargetState {
                format: surface_format,
                write_mask: wgpu::ColorWrites::COLOR,
                blend: None,
            }],
        }),
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
    });

    let max_mips = 16;

    let bloom_texture_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("bloom texture bgl"),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::StorageTexture {
                access: wgpu::StorageTextureAccess::ReadWrite,
                format: wgpu::TextureFormat::Rgba32Float,
                view_dimension: wgpu::TextureViewDimension::D2,
            },
            count: Some(core::num::NonZeroU32::new(max_mips).unwrap()),
        }],
    });

    let bloom_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("bloom pipeline layout"),
        bind_group_layouts: &[&bind_group_layout, &bloom_texture_bgl],
        push_constant_ranges: &[wgpu::PushConstantRange {
            stages: wgpu::ShaderStages::COMPUTE,
            range: 0..128,
        }],
    });

    let pre_filter_module = unsafe {
        device.create_shader_module_spirv(&wgpu::include_spirv_raw!(
            "../shaders/downsample_pre_filter.spv"
        ))
    };

    let pre_filter_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("pre filter pipeline"),
        layout: Some(&bloom_pipeline_layout),
        module: &pre_filter_module,
        entry_point: "downsample_pre_filter",
    });

    let downsample_module = unsafe {
        device.create_shader_module_spirv(&wgpu::include_spirv_raw!("../shaders/downsample.spv"))
    };

    let downsample_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("downsample pipeline"),
        layout: Some(&bloom_pipeline_layout),
        module: &downsample_module,
        entry_point: "downsample",
    });

    let upsample_module = unsafe {
        device.create_shader_module_spirv(&wgpu::include_spirv_raw!("../shaders/upsample.spv"))
    };

    let upsample_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("upsample pipeline"),
        layout: Some(&bloom_pipeline_layout),
        module: &upsample_module,
        entry_point: "upsample",
    });

    let upsample_final_module = unsafe {
        device
            .create_shader_module_spirv(&wgpu::include_spirv_raw!("../shaders/upsample_final.spv"))
    };

    let single_storage_texture_bgl =
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("single storage texture bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::StorageTexture {
                    access: wgpu::StorageTextureAccess::ReadWrite,
                    format: wgpu::TextureFormat::Rgba32Float,
                    view_dimension: wgpu::TextureViewDimension::D2,
                },
                count: None,
            }],
        });

    let upsample_final_pipeline_layout =
        device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("upsample final pipeline layout"),
            bind_group_layouts: &[&bind_group_layout, &single_storage_texture_bgl],
            push_constant_ranges: &[],
        });

    let upsample_final_pipeline =
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("upsample final pipeline"),
            layout: Some(&upsample_final_pipeline_layout),
            module: &upsample_final_module,
            entry_point: "upsample_final",
        });

    let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("sampler"),
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        ..Default::default()
    });

    let (vertices, indices, num_indices, emissive_texture) =
        load_gltf_from_bytes(include_bytes!("../bloom_example.glb"), &device, &queue)?;

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("bind group"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&emissive_texture),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(&sampler),
            },
        ],
    });

    let mut keyboard_state = KeyboardState::default();

    let mut camera = dolly::rig::CameraRig::builder()
        .with(dolly::drivers::Position::new(dolly::glam::Vec3::new(
            2.0, 4.0, 1.0,
        )))
        .with(dolly::drivers::YawPitch::new().pitch_degrees(-74.0))
        .with(dolly::drivers::Smooth::new_position_rotation(0.5, 0.25))
        .build();

    let mut cursor_grab = false;

    let mut screen_center = winit::dpi::LogicalPosition::new(
        surface_configuration.width as f64 / 2.0,
        surface_configuration.height as f64 / 2.0,
    );

    let mut perspective_matrix = ultraviolet::projection::perspective_infinite_z_wgpu_dx(
        59.0_f32.to_radians(),
        surface_configuration.width as f32 / surface_configuration.height as f32,
        0.1,
    );

    let mut depth_buffer_view = create_depth_buffer(&device, &surface_configuration);

    let mut hdr_framebuffer_view = create_hdr_framebuffer(&device, &surface_configuration);

    let mut hdr_texture_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("hdr texture bind group"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&hdr_framebuffer_view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(&sampler),
            },
        ],
    });

    let mut hdr_texture_storage_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("hdr texture storage bind group"),
        layout: &single_storage_texture_bgl,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: wgpu::BindingResource::TextureView(&hdr_framebuffer_view),
        }],
    });

    let mut bloom_mips =
        bloom_mips_for_dimensions(surface_configuration.width, surface_configuration.height);

    let mut mip_chain_a = MipChain::new(
        &device,
        &surface_configuration,
        bloom_mips,
        max_mips,
        &bloom_texture_bgl,
        &bind_group_layout,
        &sampler,
    );

    let mut mip_chain_b = MipChain::new(
        &device,
        &surface_configuration,
        bloom_mips,
        max_mips,
        &bloom_texture_bgl,
        &bind_group_layout,
        &sampler,
    );

    event_loop.run(move |event, _, control_flow| match event {
        Event::WindowEvent { event, .. } => match event {
            WindowEvent::CloseRequested => *control_flow = ControlFlow::Exit,
            WindowEvent::Resized(new_size) => {
                surface_configuration.width = new_size.width;
                surface_configuration.height = new_size.height;
                surface.configure(&device, &surface_configuration);

                screen_center = winit::dpi::LogicalPosition::new(
                    surface_configuration.width as f64 / 2.0,
                    surface_configuration.height as f64 / 2.0,
                );

                perspective_matrix = ultraviolet::projection::perspective_infinite_z_wgpu_dx(
                    59.0_f32.to_radians(),
                    surface_configuration.width as f32 / surface_configuration.height as f32,
                    0.1,
                );

                depth_buffer_view = create_depth_buffer(&device, &surface_configuration);

                hdr_framebuffer_view = create_hdr_framebuffer(&device, &surface_configuration);

                hdr_texture_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("hdr texture bind group"),
                    layout: &bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(&hdr_framebuffer_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::Sampler(&sampler),
                        },
                    ],
                });

                hdr_texture_storage_bind_group =
                    device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("hdr texture storage bind group"),
                        layout: &single_storage_texture_bgl,
                        entries: &[wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(&hdr_framebuffer_view),
                        }],
                    });

                bloom_mips = bloom_mips_for_dimensions(
                    surface_configuration.width,
                    surface_configuration.height,
                );

                mip_chain_a = MipChain::new(
                    &device,
                    &surface_configuration,
                    bloom_mips,
                    max_mips,
                    &bloom_texture_bgl,
                    &bind_group_layout,
                    &sampler,
                );

                mip_chain_b = MipChain::new(
                    &device,
                    &surface_configuration,
                    bloom_mips,
                    max_mips,
                    &bloom_texture_bgl,
                    &bind_group_layout,
                    &sampler,
                );
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
                                window.set_cursor_position(screen_center).unwrap();
                            }

                            window.set_cursor_visible(!cursor_grab);
                            window.set_cursor_grab(cursor_grab).unwrap();
                        }
                    }
                    _ => {}
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                if cursor_grab {
                    let position = position.to_logical::<f64>(window.scale_factor());

                    window.set_cursor_position(screen_center).unwrap();

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
                * dolly::glam::Vec3::new(right as f32, 0.0, -forwards as f32).clamp_length_max(1.0);

            camera
                .driver_mut::<dolly::drivers::Position>()
                .translate(move_vec * delta_time * 10.0);

            camera.update(delta_time);

            window.request_redraw();
        }
        Event::RedrawRequested(_) => {
            let texture = surface.get_current_texture().unwrap();

            let camera_view = Mat4::look_at(
                Vec3::from(<[f32; 3]>::from(camera.final_transform.position)),
                Vec3::from(<[f32; 3]>::from(
                    camera.final_transform.position + camera.final_transform.forward(),
                )),
                Vec3::from(<[f32; 3]>::from(camera.final_transform.up())),
            );

            let texture_view = texture.texture.create_view(&wgpu::TextureViewDescriptor {
                label: Some("surface texture view"),
                ..Default::default()
            });

            let mut command_encoder =
                device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("command encoder"),
                });

            let mut render_pass = command_encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("render pass"),
                color_attachments: &[wgpu::RenderPassColorAttachment {
                    view: &hdr_framebuffer_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: true,
                    },
                }],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &depth_buffer_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: true,
                    }),
                    stencil_ops: None,
                }),
            });

            render_pass.set_pipeline(&render_pipeline);
            render_pass.set_bind_group(0, &bind_group, &[]);
            render_pass.set_index_buffer(indices.slice(..), wgpu::IndexFormat::Uint32);
            render_pass.set_vertex_buffer(0, vertices.slice(..));
            render_pass.set_push_constants(
                wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                0,
                bytemuck::bytes_of(&(perspective_matrix * camera_view)),
            );
            render_pass.draw_indexed(0..num_indices, 0, 0..1);

            drop(render_pass);

            let mut compute_pass =
                command_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("compute pass"),
                });

            compute_pass.set_pipeline(&pre_filter_pipeline);
            compute_pass.set_bind_group(0, &hdr_texture_bind_group, &[]);
            compute_pass.set_bind_group(1, &mip_chain_a.storage_mips_bind_group, &[]);
            compute_pass.set_push_constants(0, bytemuck::bytes_of(&FilterConstants {
                threshold: 7.5,
                knee: 5.0,
            }));
            compute_pass.dispatch(
                dispatch_count(surface_configuration.width >> 1, 8),
                dispatch_count(surface_configuration.height >> 1, 8),
                1,
            );

            compute_pass.set_pipeline(&downsample_pipeline);

            for i in 0..bloom_mips - 1 {
                let mut src = &mip_chain_a;
                let mut dst = &mip_chain_b;

                if i % 2 == 1 {
                    std::mem::swap(&mut src, &mut dst);
                }

                compute_pass.set_bind_group(0, &src.sampled_texture_bind_group, &[]);
                compute_pass.set_bind_group(1, &dst.storage_mips_bind_group, &[]);
                compute_pass.set_push_constants(0, bytemuck::bytes_of(&(i as u32)));
                compute_pass.dispatch(
                    dispatch_count(surface_configuration.width >> (i + 2), 8),
                    dispatch_count(surface_configuration.height >> (i + 2), 8),
                    1,
                );
            }

            compute_pass.set_pipeline(&upsample_pipeline);

            for i in (0..bloom_mips - 1).rev() {
                let mut src = &mip_chain_a;
                let mut dst = &mip_chain_b;

                if i % 2 == 0 {
                    std::mem::swap(&mut src, &mut dst);
                }

                compute_pass.set_bind_group(0, &src.sampled_texture_bind_group, &[]);
                compute_pass.set_bind_group(1, &dst.storage_mips_bind_group, &[]);
                compute_pass.set_push_constants(0, bytemuck::bytes_of(&(i as u32)));
                compute_pass.dispatch(
                    dispatch_count(surface_configuration.width >> (i + 1), 8),
                    dispatch_count(surface_configuration.height >> (i + 1), 8),
                    1,
                );
            }

            compute_pass.set_pipeline(&upsample_final_pipeline);
            compute_pass.set_bind_group(0, &mip_chain_a.sampled_texture_bind_group, &[]);
            compute_pass.set_bind_group(1, &hdr_texture_storage_bind_group, &[]);
            compute_pass.dispatch(
                dispatch_count(surface_configuration.width, 8),
                dispatch_count(surface_configuration.height, 8),
                1,
            );

            drop(compute_pass);

            let mut render_pass = command_encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("tonemap render pass"),
                color_attachments: &[wgpu::RenderPassColorAttachment {
                    view: &texture_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: true,
                    },
                }],
                depth_stencil_attachment: None,
            });

            render_pass.set_pipeline(&tonemap_pipeline);
            render_pass.set_bind_group(0, &hdr_texture_bind_group, &[]);
            render_pass.set_push_constants(
                wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                0,
                bytemuck::bytes_of(&colstodian::tonemap::BakedLottesTonemapperParams::from(
                    colstodian::tonemap::LottesTonemapperParams {
                        ..Default::default()
                    },
                )),
            );
            render_pass.draw(0..3, 0..1);

            drop(render_pass);

            queue.submit(std::iter::once(command_encoder.finish()));

            texture.present();
        }
        _ => {}
    });
}

fn create_depth_buffer(
    device: &wgpu::Device,
    surface_configuration: &wgpu::SurfaceConfiguration,
) -> wgpu::TextureView {
    device
        .create_texture(&wgpu::TextureDescriptor {
            label: Some("depth buffer"),
            size: wgpu::Extent3d {
                width: surface_configuration.width,
                height: surface_configuration.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        })
        .create_view(&Default::default())
}

fn create_hdr_framebuffer(
    device: &wgpu::Device,
    surface_configuration: &wgpu::SurfaceConfiguration,
) -> wgpu::TextureView {
    device
        .create_texture(&wgpu::TextureDescriptor {
            label: Some("hdr framebuffer"),
            size: wgpu::Extent3d {
                width: surface_configuration.width,
                height: surface_configuration.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::STORAGE_BINDING,
        })
        .create_view(&Default::default())
}

struct MipChain {
    storage_mips_bind_group: wgpu::BindGroup,
    sampled_texture_bind_group: wgpu::BindGroup,
}

impl MipChain {
    fn new(
        device: &wgpu::Device,
        surface_configuration: &wgpu::SurfaceConfiguration,
        mip_levels: u32,
        max_mips: u32,
        bloom_texture_bgl: &wgpu::BindGroupLayout,
        sampled_texture_bgl: &wgpu::BindGroupLayout,
        sampler: &wgpu::Sampler,
    ) -> Self {
        let width = (surface_configuration.width / 2).max(1);
        let height = (surface_configuration.height / 2).max(1);

        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("bloom texture"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: mip_levels,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba32Float,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
        });

        let dummy_mips = std::iter::repeat(mip_levels - 1).take((max_mips - mip_levels) as usize);

        let views: Vec<_> = (0..mip_levels)
            .map(|i| {
                texture.create_view(&wgpu::TextureViewDescriptor {
                    base_mip_level: i,
                    mip_level_count: Some(core::num::NonZeroU32::new(1).unwrap()),
                    ..Default::default()
                })
            })
            .collect();

        let view_refs: Vec<_> = (0..mip_levels).chain(dummy_mips).map(|i| &views[i as usize]).collect();

        Self {
            storage_mips_bind_group: device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("bloom storage mips bind group"),
                layout: bloom_texture_bgl,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureViewArray(&view_refs),
                }],
            }),
            sampled_texture_bind_group: device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("bloom sample texture bind group"),
                layout: sampled_texture_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&texture.create_view(&Default::default())),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(sampler),
                    },
                ],
            }),
        }
    }
}

#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
struct FilterConstants {
    threshold: f32,
    knee: f32,
}

#[derive(Default)]
struct KeyboardState {
    forwards: bool,
    right: bool,
    backwards: bool,
    left: bool,
}

#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
struct Vertex {
    position: Vec3,
    uv: Vec2,
}

fn load_gltf_from_bytes(
    bytes: &[u8],
    device: &wgpu::Device,
    queue: &wgpu::Queue,
) -> anyhow::Result<(wgpu::Buffer, wgpu::Buffer, u32, wgpu::TextureView)> {
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
        device,
        queue,
    )?;

    let num_indices = indices.len() as u32;

    let vertices = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("vertices"),
        contents: bytemuck::cast_slice(&vertices),
        usage: wgpu::BufferUsages::VERTEX,
    });

    let indices = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("indices"),
        contents: bytemuck::cast_slice(&indices),
        usage: wgpu::BufferUsages::INDEX,
    });

    Ok((vertices, indices, num_indices, texture))
}

fn load_texture_from_gltf(
    texture: gltf::texture::Texture,
    label: &str,
    buffer_blob: &[u8],
    device: &wgpu::Device,
    queue: &wgpu::Queue,
) -> anyhow::Result<wgpu::TextureView> {
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

    Ok(device
        .create_texture_with_data(
            queue,
            &wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d {
                    width: decoded_rgba8.width(),
                    height: decoded_rgba8.height(),
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8UnormSrgb,
                usage: wgpu::TextureUsages::TEXTURE_BINDING,
            },
            &*decoded_rgba8,
        )
        .create_view(&Default::default()))
}

// https://github.com/bevyengine/bevy/blob/2c11ca0291f94b14ee32883d40f8243f3c8e3d6c/pipelined/bevy_hdr/src/bloom_node.rs#L292-L297
fn bloom_mips_for_dimensions(width: u32, height: u32) -> u32 {
    let min_element = width.min(height) / 2;
    let mut mips = 1;

    while (min_element >> mips) > 0 {
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
