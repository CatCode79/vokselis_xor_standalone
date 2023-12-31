pub(crate) struct HdrBackBuffer {
    pub(crate) render_bind_group: wgpu::BindGroup,
    pub(crate) storage_bind_group: wgpu::BindGroup,
}

impl HdrBackBuffer {
    pub(crate) const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
    pub(crate) const DEFAULT_RESOLUTION: (u32, u32) = (1280, 720);
    pub(crate) const DESC_COMPUTE: wgpu::BindGroupLayoutDescriptor<'static> =
        wgpu::BindGroupLayoutDescriptor {
            label: Some("Storage Texture Layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::StorageTexture {
                    access: wgpu::StorageTextureAccess::WriteOnly,
                    format: Self::FORMAT,
                    view_dimension: wgpu::TextureViewDimension::D2,
                },
                count: None,
            }],
        };
    pub(crate) const DESC_RENDER: wgpu::BindGroupLayoutDescriptor<'static> =
        wgpu::BindGroupLayoutDescriptor {
            label: Some("BackBuffer: Render Bind Group Layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            }],
        };

    pub(crate) fn new(device: &wgpu::Device, (width, height): (u32, u32)) -> Self {
        let texture_view = {
            let size = wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            };
            let texture = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("Texture: HdrBackbuffer"),
                size,
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: Self::FORMAT,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::STORAGE_BINDING
                    | wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            });
            texture.create_view(&Default::default())
        };

        let binding_resource = &[wgpu::BindGroupEntry {
            binding: 0,
            resource: wgpu::BindingResource::TextureView(&texture_view),
        }];
        let render_bind_group = {
            let render_bind_group_layout = device.create_bind_group_layout(&Self::DESC_RENDER);
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("BackBuffer: Render Bind Group"),
                layout: &render_bind_group_layout,
                entries: binding_resource,
            })
        };
        let storage_bind_group = {
            let storage_bind_group_layout = device.create_bind_group_layout(&Self::DESC_COMPUTE);
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("BackBuffer: Render Bind Group"),
                layout: &storage_bind_group_layout,
                entries: binding_resource,
            })
        };

        Self {
            render_bind_group,
            storage_bind_group,
        }
    }
}
