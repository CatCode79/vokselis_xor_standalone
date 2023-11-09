mod global_ubo;
mod hdr_backbuffer;

pub(crate) use global_ubo::GlobalUniformBinding;
pub(crate) use global_ubo::Uniform;
pub(crate) use hdr_backbuffer::HdrBackBuffer;

use crate::pipelines::present_pipeline::PresentPipeline;
use crate::utils::frame_counter::FrameCounter;
use crate::utils::input::Input;
use crate::{Camera, CameraBinding};

use wgpu::StoreOp;
use winit::{dpi::PhysicalSize, window::Window};

use std::{sync::Arc, time::Instant};

pub(crate) struct Context {
    #[allow(dead_code)]
    adapter: wgpu::Adapter,
    pub(crate) device: Arc<wgpu::Device>,
    pub(crate) queue: wgpu::Queue,
    surface: wgpu::Surface,
    pub(crate) surface_config: wgpu::SurfaceConfiguration,

    pub(crate) camera: Camera,
    pub(crate) camera_binding: CameraBinding,

    pub(crate) render_backbuffer: HdrBackBuffer,

    rgb_texture: wgpu::Texture,

    pub(crate) width: u32,
    pub(crate) height: u32,

    timeline: Instant,

    pub(crate) global_uniform: Uniform,
    pub(crate) global_uniform_binding: GlobalUniformBinding,

    present_pipeline: PresentPipeline,
}

impl Context {
    /// Create a new window with a given `window`
    pub(crate) async fn new(window: &Window, camera: Camera) -> Result<Self, String> {
        // Create new instance using first-tier backend of WGPU
        // One of Vulkan + Metal + DX12 + Browser WebGPU
        let instance_desc = wgpu::InstanceDescriptor {
            backends: wgpu::Backends::PRIMARY,
            ..Default::default()
        };
        let instance = wgpu::Instance::new(instance_desc);

        // Create a `surface` represents a platform-specific window
        // onto which rendered images may be presented
        let surface = unsafe { instance.create_surface(&window) }.unwrap();

        // Get a handle to a physical device
        let adapter: wgpu::Adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: Some(&surface),
            })
            .await
            .ok_or("Failed to create device adapter.".to_string())?;

        // Use default features and limits for your machine
        let features = adapter.features();
        let limits = adapter.limits();
        let surface_format = wgpu::TextureFormat::Bgra8Unorm;

        // Create the logical device and command queue
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("Device Descriptor"),
                    features,
                    limits: limits.clone(),
                },
                None,
            )
            .await
            .map_err(|e| e.to_string())?;
        let device = Arc::new(device);

        let PhysicalSize { width, height } = window.inner_size();
        let surface_config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width,
            height,
            present_mode: wgpu::PresentMode::Fifo,
            alpha_mode: wgpu::CompositeAlphaMode::Auto,
            view_formats: vec![],
        };
        surface.configure(&device, &surface_config);

        let render_backbuffer = HdrBackBuffer::new(&device, HdrBackBuffer::DEFAULT_RESOLUTION);
        let rgb_texture = create_rgb_framebuffer(&device, &surface_config);

        let present_shader = wgpu::include_wgsl!("../shaders/present.wgsl");
        let present_pipeline = PresentPipeline::new(&device, surface_format, present_shader);

        Ok(Self {
            camera,
            camera_binding: CameraBinding::new(&device),

            rgb_texture,

            render_backbuffer,

            width,
            height,

            timeline: Instant::now(),

            present_pipeline,

            global_uniform: Uniform::default(),
            global_uniform_binding: GlobalUniformBinding::new(&device),

            device,
            adapter,
            queue,
            surface,
            surface_config,
        })
    }

    pub(crate) fn update(&mut self, frame_counter: &FrameCounter, input: &Input) {
        self.global_uniform.time = self.timeline.elapsed().as_secs_f32();
        self.global_uniform.time_delta = frame_counter.time_delta();
        self.global_uniform.frame = frame_counter.frame_count;
        self.global_uniform.resolution = [self.width as _, self.height as _];
        input.process_position(&mut self.global_uniform);

        self.global_uniform_binding
            .update(&self.queue, &self.global_uniform);

        self.camera_binding.update(&self.queue, &mut self.camera);
    }

    pub(crate) fn resize(&mut self, width: u32, height: u32) {
        self.width = width;
        self.height = height;
        self.surface_config.height = height;
        self.surface_config.width = width;
        self.surface.configure(&self.device, &self.surface_config);

        self.rgb_texture = create_rgb_framebuffer(&self.device, &self.surface_config);

        self.camera.set_aspect(width, height);
    }

    pub(crate) fn render(&self) -> Result<(), wgpu::SurfaceError> {
        let frame = self.surface.get_current_texture()?;
        let frame_view = frame.texture.create_view(&Default::default());

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Present Encoder"),
            });

        let rgb = self.rgb_texture.create_view(&Default::default());
        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Present Pass"),
            color_attachments: &[
                Some(wgpu::RenderPassColorAttachment {
                    view: &frame_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: StoreOp::Store,
                    },
                }),
                Some(wgpu::RenderPassColorAttachment {
                    view: &rgb,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: StoreOp::Store,
                    },
                }),
            ],
            ..Default::default()
        });

        self.present_pipeline.record(
            &mut rpass,
            &self.global_uniform_binding,
            &self.render_backbuffer.render_bind_group,
        );
        drop(rpass);

        self.queue.submit(Some(encoder.finish()));

        frame.present();

        Ok(())
    }
}

#[derive(Debug)]
pub(crate) struct RendererInfo {
    pub(crate) device_name: String,
    pub(crate) device_type: String,
    pub(crate) vendor_name: String,
    pub(crate) backend: String,
    pub(crate) screen_format: wgpu::TextureFormat,
}

impl std::fmt::Display for RendererInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Vendor name: {}", self.vendor_name)?;
        writeln!(f, "Device name: {}", self.device_name)?;
        writeln!(f, "Device type: {}", self.device_type)?;
        writeln!(f, "Backend: {}", self.backend)?;
        write!(f, "Screen format: {:?}", self.screen_format)?;
        Ok(())
    }
}

fn create_rgb_framebuffer(
    device: &wgpu::Device,
    config: &wgpu::SurfaceConfiguration,
) -> wgpu::Texture {
    let size = wgpu::Extent3d {
        width: config.width,
        height: config.height,
        depth_or_array_layers: 1,
    };
    let multisampled_frame_descriptor = &wgpu::TextureDescriptor {
        label: Some("RGB Texture"),
        format: wgpu::TextureFormat::Rgba8Unorm,
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    };

    device.create_texture(multisampled_frame_descriptor)
}
