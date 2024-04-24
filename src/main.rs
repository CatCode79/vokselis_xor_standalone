#![warn(
absolute_paths_not_starting_with_crate,
//box_pointers,
elided_lifetimes_in_paths,
explicit_outlives_requirements,
keyword_idents,
let_underscore_drop,
macro_use_extern_crate,
meta_variable_misuse,
missing_abi,
//missing_copy_implementations,
//missing_debug_implementations,
//missing_docs,
non_ascii_idents,
noop_method_call,
pointer_structural_match,
rust_2021_incompatible_closure_captures,
rust_2021_incompatible_or_patterns,
rust_2021_prefixes_incompatible_syntax,
rust_2021_prelude_collisions,
single_use_lifetimes,
trivial_casts,
trivial_numeric_casts,
unreachable_pub,
//unsafe_code,
unsafe_op_in_unsafe_fn,
unstable_features,
unused_crate_dependencies,
unused_extern_crates,
unused_import_braces,
unused_lifetimes,
unused_macro_rules,
unused_qualifications,
//unused_results,
dead_code,
variant_size_differences,
clippy::cargo,
clippy::complexity,
clippy::correctness,
clippy::nursery,
//clippy::pedantic,
clippy::perf,
clippy::restriction,
clippy::style,
clippy::suspicious,
)]

mod camera;
mod context;
mod pipelines;
mod utils;

use crate::camera::{Camera, CameraBinding};
use crate::context::{Context, GlobalUniformBinding, HdrBackBuffer, Uniform};
use crate::pipelines::{raycast, xor_compute};
use crate::utils::{dispatch_optimal, NonZeroSized};

use bytemuck::{Pod, Zeroable};
use pollster::FutureExt;
use utils::{frame_counter::FrameCounter, input::Input};
use winit::{
    dpi::{LogicalSize, PhysicalPosition, PhysicalSize},
    event::{DeviceEvent, ElementState, Event, KeyEvent, MouseScrollDelta, WindowEvent},
    event_loop::{ControlFlow, EventLoopBuilder},
    keyboard::{Key, NamedKey},
    window::WindowBuilder,
};

use std::sync::Arc;
use std::path::PathBuf;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct TimestampData {
    start: u64,
    end: u64,
}

struct Xor {
    xor_texture: xor_compute::XorCompute,
    raycast_single: raycast::RaycastPipeline,

    timestamp: wgpu::QuerySet,
    timestamp_period: f32,
    timestamp_buffer: wgpu::Buffer,
}

impl Xor {
    fn init(ctx: &mut Context) -> Self {
        let raycast_single = {
            let module_desc = wgpu::include_wgsl!("../shaders/raycast_compute.wgsl");
            pipelines::raycast::RaycastPipeline::new(&ctx.device, module_desc.clone(), "single")
        };

        let xor_texture = {
            let shader_module_desc = wgpu::include_wgsl!("../shaders/xor.wgsl");
            pipelines::xor_compute::XorCompute::new(&ctx.device, shader_module_desc)
        };

        let timestamp = ctx.device.create_query_set(&wgpu::QuerySetDescriptor {
            label: None,
            count: 2,
            ty: wgpu::QueryType::Timestamp,
        });
        let timestamp_period = ctx.queue.get_timestamp_period();
        let timestamp_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Query Buffer"),
            size: std::mem::size_of::<TimestampData>() as _,
            usage: wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::MAP_READ
                | wgpu::BufferUsages::QUERY_RESOLVE,
            mapped_at_creation: false,
        });

        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("XOR Update encoder"),
            });

        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("XOR Update Pass"),
            ..Default::default()
        });
        xor_texture.record(&mut cpass, &ctx.global_uniform_binding);
        drop(cpass);
        ctx.queue.submit(Some(encoder.finish()));

        Self {
            xor_texture,
            raycast_single,

            timestamp,
            timestamp_period,
            timestamp_buffer,
        }
    }

    fn update(&mut self, ctx: &mut Context) {
        if ctx.global_uniform.frame % 100 == 0 {
            let _ = self
                .timestamp_buffer
                .slice(..)
                .map_async(wgpu::MapMode::Read, |_| ());
            {
                ctx.device.poll(wgpu::Maintain::Wait);
                let timestamp_view = self
                    .timestamp_buffer
                    .slice(..std::mem::size_of::<TimestampData>() as wgpu::BufferAddress)
                    .get_mapped_range();
                let timestamp_data: &TimestampData = bytemuck::from_bytes(&*timestamp_view);
                let nanoseconds =
                    (timestamp_data.end - timestamp_data.start) as f32 * self.timestamp_period;
                let time_period = std::time::Duration::from_nanos(nanoseconds as _);
                eprintln!("Time on raycast shader: {:?} (single pass)", time_period);
            }
            self.timestamp_buffer.unmap();
        }
    }

    fn render(&mut self, ctx: &Context) {
        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Volume Encoder"),
            });

        encoder.write_timestamp(&self.timestamp, 0);

        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Raycast Pass"),
            ..Default::default()
        });

        cpass.set_pipeline(&self.raycast_single.pipeline);

        cpass.set_bind_group(0, &ctx.global_uniform_binding.binding, &[]);
        cpass.set_bind_group(1, &ctx.camera_binding.bind_group, &[]);
        cpass.set_bind_group(2, &self.xor_texture.storage_bind_group, &[]);
        cpass.set_bind_group(3, &ctx.render_backbuffer.storage_bind_group, &[]);
        let (width, height) = HdrBackBuffer::DEFAULT_RESOLUTION;
        cpass.dispatch_workgroups(dispatch_optimal(width, 8), dispatch_optimal(height, 8), 1);
        drop(cpass);

        encoder.write_timestamp(&self.timestamp, 1);
        encoder.resolve_query_set(&self.timestamp, 0..2, &self.timestamp_buffer, 0);

        ctx.queue.submit(Some(encoder.finish()));
    }
}

fn main() -> Result<(), String> {
    let event_loop = EventLoopBuilder::<(PathBuf, wgpu::ShaderModule)>::with_user_event()
        .build()
        .map_err(|e| e.to_string())?;
    let window = Arc::new(WindowBuilder::new()
        .with_title("Vokselis")
        .with_inner_size(LogicalSize::new(1280, 720))
        .build(&event_loop)
        .map_err(|e| e.to_string())?);
    let window_size = window.inner_size();

    let camera = Camera::new(
        3.,
        -0.5,
        1.,
        (0., 0., 0.).into(),
        window_size.width as f32 / window_size.height as f32,
    );

    env_logger::init();

    let mut context = Context::new(window.clone(), camera).block_on()?;

    let mut frame_counter = FrameCounter::new();
    let mut input = Input::new();

    let mut mouse_dragged = false;
    let rotate_speed = 0.0025;
    let zoom_speed = 0.002;

    let mut xor = Xor::init(&mut context);

    let mut main_window_focused = false;
    event_loop
        .run(move |event, target| {
            target.set_control_flow(ControlFlow::Wait);

            match event {
                Event::AboutToWait => {
                    context.update(&frame_counter, &input);
                    xor.update(&mut context);
                    window.request_redraw();
                }

                Event::WindowEvent {
                    event: window_event,
                    window_id,
                    ..
                } if window.id() == window_id => {
                    input.update(&window_event, &window);

                    match window_event {
                        WindowEvent::Focused(focused) => main_window_focused = focused,

                        WindowEvent::CloseRequested
                        | WindowEvent::KeyboardInput {
                            event:
                                KeyEvent {
                                    logical_key: Key::Named(NamedKey::Escape),
                                    ..
                                },
                            ..
                        } => target.exit(),

                        WindowEvent::RedrawRequested => {
                            frame_counter.record();

                            xor.render(&context);

                            match context.render() {
                                Ok(_) => {}
                                Err(wgpu::SurfaceError::Lost) => {
                                    context.resize(context.width, context.height);
                                    window.request_redraw();
                                }
                                Err(wgpu::SurfaceError::OutOfMemory) => target.exit(),
                                Err(e) => {
                                    eprintln!("{:?}", e);
                                    window.request_redraw();
                                }
                            }
                        }

                        WindowEvent::Resized(PhysicalSize { width, height }) => {
                            if width != 0 && height != 0 {
                                context.resize(width, height);
                            }
                        }

                        _ => {}
                    }
                }

                Event::DeviceEvent { ref event, .. } if main_window_focused => match event {
                    DeviceEvent::Button {
                        button: 1,
                        state: statee,
                    } => {
                        let is_pressed = *statee == ElementState::Pressed;
                        mouse_dragged = is_pressed;
                    }
                    DeviceEvent::MouseWheel { delta, .. } => {
                        let scroll_amount = -match delta {
                            MouseScrollDelta::LineDelta(_, scroll) => scroll * 1.0,
                            MouseScrollDelta::PixelDelta(PhysicalPosition {
                                y: scroll, ..
                            }) => *scroll as f32,
                        };
                        context.camera.add_zoom(scroll_amount * zoom_speed);
                    }
                    DeviceEvent::MouseMotion { delta } => {
                        if mouse_dragged {
                            context.camera.add_yaw(-delta.0 as f32 * rotate_speed);
                            context.camera.add_pitch(delta.1 as f32 * rotate_speed);
                        }
                    }
                    _ => (),
                },

                _ => {}
            }
        })
        .map_err(|e| e.to_string())
}
