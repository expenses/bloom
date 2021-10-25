use ash::vk;
pub use rendering_shaders::FilterConstants;
use std::ffi::CStr;

pub struct ComputePipelines {
    pub downsample_initial: vk::Pipeline,
    pub downsample: vk::Pipeline,
    pub upsample: vk::Pipeline,
    pub upsample_final: vk::Pipeline,
    pub upsample_final_pipeline_layout: vk::PipelineLayout,
    pub bloom_pipeline_layout: vk::PipelineLayout,
}

impl ComputePipelines {
    pub fn new(
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
                        .size(std::mem::size_of::<FilterConstants>() as u32)]),
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

        let downsample_initial_stage = load_shader_module_as_stage(
            include_bytes!("../../shaders/downsample_initial.spv"),
            vk::ShaderStageFlags::COMPUTE,
            device,
            b"downsample_initial\0",
        )?;

        let downsample_initial_create_info = vk::ComputePipelineCreateInfo::builder()
            .stage(*downsample_initial_stage)
            .layout(bloom_pipeline_layout);

        let downsample_stage = load_shader_module_as_stage(
            include_bytes!("../../shaders/downsample.spv"),
            vk::ShaderStageFlags::COMPUTE,
            device,
            b"downsample\0",
        )?;

        let downsample_create_info = vk::ComputePipelineCreateInfo::builder()
            .stage(*downsample_stage)
            .layout(bloom_pipeline_layout);

        let upsample_stage = load_shader_module_as_stage(
            include_bytes!("../../shaders/upsample.spv"),
            vk::ShaderStageFlags::COMPUTE,
            device,
            b"upsample\0",
        )?;

        let upsample_create_info = vk::ComputePipelineCreateInfo::builder()
            .stage(*upsample_stage)
            .layout(bloom_pipeline_layout);

        let upsample_final_stage = load_shader_module_as_stage(
            include_bytes!("../../shaders/upsample_final.spv"),
            vk::ShaderStageFlags::COMPUTE,
            device,
            b"upsample_final\0",
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

    pub fn destroy(&self, device: &ash::Device) {
        unsafe {
            device.destroy_pipeline(self.downsample_initial, None);
            device.destroy_pipeline(self.downsample, None);
            device.destroy_pipeline(self.upsample, None);
            device.destroy_pipeline(self.upsample_final, None);
            device.destroy_pipeline_layout(self.bloom_pipeline_layout, None);
            device.destroy_pipeline_layout(self.upsample_final_pipeline_layout, None);
        }
    }
}

fn load_shader_module_as_stage<'a>(
    bytes: &[u8],
    stage: vk::ShaderStageFlags,
    device: &ash::Device,
    entry_point_with_nul: &'a [u8],
) -> anyhow::Result<vk::PipelineShaderStageCreateInfoBuilder<'a>> {
    let spv = ash::util::read_spv(&mut std::io::Cursor::new(bytes))?;
    let module = unsafe {
        device.create_shader_module(&vk::ShaderModuleCreateInfo::builder().code(&spv), None)
    }?;

    Ok(vk::PipelineShaderStageCreateInfo::builder()
        .module(module)
        .stage(stage)
        .name(CStr::from_bytes_with_nul(entry_point_with_nul)?))
}

pub struct DescriptorSetLayouts {
    pub sampled_texture: vk::DescriptorSetLayout,
    pub bloom_texture: vk::DescriptorSetLayout,
    pub storage_texture: vk::DescriptorSetLayout,
}

impl DescriptorSetLayouts {
    pub fn new(device: &ash::Device, max_mips: u32) -> anyhow::Result<Self> {
        unsafe {
            Ok(Self {
                sampled_texture: device.create_descriptor_set_layout(
                    &*vk::DescriptorSetLayoutCreateInfo::builder().bindings(&[
                        *vk::DescriptorSetLayoutBinding::builder()
                            .binding(0)
                            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                            .descriptor_count(1)
                            .stage_flags(vk::ShaderStageFlags::COMPUTE),
                    ]),
                    None,
                )?,
                bloom_texture: device.create_descriptor_set_layout(
                    &*vk::DescriptorSetLayoutCreateInfo::builder().bindings(&[
                        *vk::DescriptorSetLayoutBinding::builder()
                            .binding(0)
                            .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                            .descriptor_count(max_mips)
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

/// Run the initial downsample and prefilter step.
///
/// Takes an descriptor set containing a sampled HDR image created from the `DescriptorSetLayouts::sampled_texture` descriptor set layout.
///
/// This is seperate from the main bloom stage as it requires the HDR image to be sampled while `compute_bloom` requires the output to be a storage image. If you're using the same HDR image for both, you'll need to transition it between the two steps.
pub unsafe fn prefilter_bloom<I>(
    device: &ash::Device,
    command_buffer: vk::CommandBuffer,
    extent: vk::Extent2D,
    bloom_texture: &BloomTextureWithMips<I>,
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

pub unsafe fn compute_bloom<I: Image>(
    device: &ash::Device,
    command_buffer: vk::CommandBuffer,
    extent: vk::Extent2D,
    bloom_mips: u32,
    bloom_texture: &BloomTextureWithMips<I>,
    compute_pipelines: &ComputePipelines,
    output_descriptor_set: vk::DescriptorSet,
) {
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

        // Insert an image barrier on the mip that was written to.
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
                image: bloom_texture.image.vk_image(),
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
                image: bloom_texture.image.vk_image(),
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

const fn dispatch_count(num: u32, group_size: u32) -> u32 {
    let mut count = num / group_size;
    let rem = num % group_size;
    if rem != 0 {
        count += 1;
    }

    count
}

pub trait Image {
    fn vk_image(&self) -> vk::Image;
    fn vk_image_view(&self) -> vk::ImageView;
}

pub struct BloomTextureWithMips<I> {
    pub image: I,
    pub views: Vec<vk::ImageView>,
    pub storage_mips_descriptor_set: vk::DescriptorSet,
    pub sampled_texture_descriptor_set: vk::DescriptorSet,
    pub sampler: vk::Sampler,
}

impl<I: Image> BloomTextureWithMips<I> {
    pub fn new(
        image: I,
        device: &ash::Device,
        mip_levels: u32,
        max_mips: u32,
        descriptor_set_layouts: &DescriptorSetLayouts,
        descriptor_pool: vk::DescriptorPool,
    ) -> anyhow::Result<Self> {
        let descriptor_sets = unsafe {
            device.allocate_descriptor_sets(
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

        Self::new_with_existing_sets(
            image,
            device,
            mip_levels,
            max_mips,
            storage_mips_descriptor_set,
            sampled_texture_descriptor_set,
        )
    }

    pub fn new_with_existing_sets(
        image: I,
        device: &ash::Device,
        mip_levels: u32,
        max_mips: u32,
        storage_mips_descriptor_set: vk::DescriptorSet,
        sampled_texture_descriptor_set: vk::DescriptorSet,
    ) -> anyhow::Result<Self> {
        let sampler = unsafe {
            device.create_sampler(
                &vk::SamplerCreateInfo::builder()
                    .mag_filter(vk::Filter::LINEAR)
                    .min_filter(vk::Filter::LINEAR)
                    .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .max_lod(vk::LOD_CLAMP_NONE),
                None,
            )
        }?;

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
                            .image(image.vk_image())
                            .view_type(vk::ImageViewType::TYPE_2D)
                            .format(vk::Format::R16G16B16A16_SFLOAT)
                            .subresource_range(subresource_range),
                        None,
                    )
                }
            })
            .collect::<Result<Vec<_>, _>>()?;

        let dummy_mips = std::iter::repeat(mip_levels - 1).take((max_mips - mip_levels) as usize);

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
                        .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                        .image_info(&[*vk::DescriptorImageInfo::builder()
                            .image_view(image.vk_image_view())
                            .image_layout(vk::ImageLayout::GENERAL)
                            .sampler(sampler)]),
                ],
                &[],
            );
        }

        Ok(Self {
            image,
            views,
            storage_mips_descriptor_set,
            sampled_texture_descriptor_set,
            sampler,
        })
    }

    pub fn cleanup(&self, device: &ash::Device) {
        for view in &self.views {
            unsafe {
                device.destroy_image_view(*view, None);
            }
        }

        unsafe { device.destroy_sampler(self.sampler, None) }
    }
}

pub fn bloom_mips_for_dimensions(width: u32, height: u32) -> u32 {
    let mut mips = 1;

    while (width.min(height) >> (mips + 1)) > 0 {
        mips += 1;
    }

    mips
}
