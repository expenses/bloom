use winit::event::{Event, WindowEvent};
use winit::event_loop::ControlFlow;

fn main() -> anyhow::Result<()> {
    let instance = wgpu::Instance::new(wgpu::Backends::PRIMARY);

    let event_loop = winit::event_loop::EventLoop::new();
    let window = winit::window::Window::new(&event_loop)?;

    let surface = unsafe {
        instance.create_surface(&window)
    };

    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: Some(&surface),
        ..Default::default()
    })).ok_or_else(|| anyhow::anyhow!("Failed to get adapter"))?;

    let surface_format = surface.get_preferred_format(&adapter).expect("unreachable; adapter was chosen to be compatible with the surface");

    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("device"),
        features: wgpu::Features::PUSH_CONSTANTS,
        limits: wgpu::Limits {
            max_push_constant_size: 128,
            ..Default::default()
        }
    }, None))?;

    let window_size = window.inner_size();

    let mut surface_configuration = wgpu::SurfaceConfiguration {
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        format: surface_format,
        width: window_size.width,
        height: window_size.height,
        present_mode: wgpu::PresentMode::Fifo
    };

    surface.configure(&device, &surface_configuration);

    event_loop.run(move |event, _, control_flow| {
        match event {
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
            },
            Event::RedrawRequested(_) => {
                let texture = surface.get_current_texture().unwrap();

                let texture_view = texture.texture.create_view(&wgpu::TextureViewDescriptor {
                    label: Some("surface texture view"),
                    ..Default::default()
                });

                let mut command_encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("command encoder")
                });

                let render_pass = command_encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("render pass"),
                    color_attachments: &[wgpu::RenderPassColorAttachment {
                        view: &texture_view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::RED),
                            store: true,
                        }
                    }],
                    depth_stencil_attachment: None,
                });

                drop(render_pass);

                queue.submit(std::iter::once(command_encoder.finish()));

                texture.present();
            },
            _ => {}
        }
    });
}
