use ultraviolet::{Vec2, Vec3, Mat4};
use wgpu::util::DeviceExt;
use winit::event::{Event, WindowEvent};
use winit::event_loop::ControlFlow;

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
            features: wgpu::Features::PUSH_CONSTANTS,
            limits: wgpu::Limits {
                max_push_constant_size: 128,
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
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler {
                    filtering: true,
                    comparison: false,
                },
                count: None,
            },
        ],
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("render pipeline layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[wgpu::PushConstantRange {
            stages: wgpu::ShaderStages::VERTEX,
            range: 0..128,
        }],
    });

    let vertex_module = device.create_shader_module(&wgpu::include_spirv!("../shaders/vertex.spv"));
    let fragment_module = device.create_shader_module(&wgpu::include_spirv!("../shaders/fragment.spv"));

    let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("sampler"),
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        ..Default::default()
    });

    let (vertices, indices, emissive_texture) =
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

    event_loop.run(move |event, _, control_flow| match event {
        Event::WindowEvent { event, .. } => match event {
            WindowEvent::CloseRequested => *control_flow = ControlFlow::Exit,
            WindowEvent::Resized(new_size) => {
                surface_configuration.width = new_size.width;
                surface_configuration.height = new_size.height;
                surface.configure(&device, &surface_configuration);
            }
            _ => {}
        },
        Event::MainEventsCleared => {
            window.request_redraw();
        }
        Event::RedrawRequested(_) => {
            let texture = surface.get_current_texture().unwrap();

            let texture_view = texture.texture.create_view(&wgpu::TextureViewDescriptor {
                label: Some("surface texture view"),
                ..Default::default()
            });

            let mut command_encoder =
                device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("command encoder"),
                });

            let render_pass = command_encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("render pass"),
                color_attachments: &[wgpu::RenderPassColorAttachment {
                    view: &texture_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::RED),
                        store: true,
                    },
                }],
                depth_stencil_attachment: None,
            });

            drop(render_pass);

            queue.submit(std::iter::once(command_encoder.finish()));

            texture.present();
        }
        _ => {}
    });
}

#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
struct RenderPushConstants {
    combined_matrix: Mat4,
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
) -> anyhow::Result<(Vec<Vertex>, Vec<u32>, wgpu::TextureView)> {
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

    Ok((vertices, indices, texture))
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
        .create_view(&wgpu::TextureViewDescriptor::default()))
}
