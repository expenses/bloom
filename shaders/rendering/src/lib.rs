#![cfg_attr(
    target_arch = "spirv",
    feature(register_attr),
    register_attr(spirv),
    no_std
)]

// This needs to be here to provide `#[panic_handler]`.
extern crate spirv_std;

use spirv_std::glam::{IVec2, IVec3, Mat4, UVec2, UVec3, Vec2, Vec3, Vec4};
use spirv_std::num_traits::Float;
use spirv_std::{Image, RuntimeArray, Sampler};

#[spirv(vertex)]
pub fn vertex(
    position: Vec3,
    uv: Vec2,
    fragment_uv: &mut Vec2,
    #[spirv(push_constant)] combined_matrix: &Mat4,
    #[spirv(position)] builtin_pos: &mut Vec4,
) {
    *fragment_uv = uv;
    *builtin_pos = *combined_matrix * position.extend(1.0);
}

#[spirv(fragment)]
pub fn fragment(
    uv: Vec2,
    #[spirv(descriptor_set = 0, binding = 0)] texture: &Image!(2D, type=f32, sampled),
    #[spirv(descriptor_set = 0, binding = 1)] sampler: &Sampler,
    output: &mut Vec4,
) {
    let emission_strength = 10.0;

    let sample: Vec4 = texture.sample(*sampler, uv);

    *output = (sample.truncate() * emission_strength).extend(1.0);
}

#[spirv(fragment)]
pub fn tonemap(
    uv: Vec2,
    #[spirv(descriptor_set = 0, binding = 0)] texture: &Image!(2D, type=f32, sampled),
    #[spirv(descriptor_set = 0, binding = 1)] sampler: &Sampler,
    #[spirv(push_constant)] params: &BakedLottesTonemapperParams,
    output: &mut Vec4,
) {
    let sample: Vec4 = texture.sample(*sampler, uv);

    *output = LottesTonemapper
        .tonemap(sample.truncate(), *params)
        .extend(1.0);
}

#[spirv(vertex)]
pub fn fullscreen_tri(
    #[spirv(vertex_index)] vert_idx: i32,
    uv: &mut Vec2,
    #[spirv(position)] builtin_pos: &mut Vec4,
) {
    // Create a "full screen triangle" by mapping the vertex index.
    // ported from https://www.saschawillems.de/blog/2016/08/13/vulkan-tutorial-on-rendering-a-fullscreen-quad-without-buffers/
    *uv = Vec2::new(((vert_idx << 1) & 2) as f32, (vert_idx & 2) as f32);
    let pos = 2.0 * *uv - Vec2::ONE;

    *builtin_pos = pos.extend(0.0).extend(1.0);

    // Flipped on Y for webgpu.
    uv.y = 1.0 - uv.y;
}

// This is just lifted from
// https://github.com/termhn/colstodian/blob/f2fb0f55d94644dbb753edd5c01da9a08f0e2d3f/src/tonemap.rs#L187-L220
// because rust-gpu support is hard.

struct LottesTonemapper;

impl LottesTonemapper {
    #[inline]
    fn tonemap_inner(x: f32, params: BakedLottesTonemapperParams) -> f32 {
        let z = x.powf(params.a);
        z / (z.powf(params.d) * params.b + params.c)
    }

    fn tonemap(&self, color: Vec3, params: BakedLottesTonemapperParams) -> Vec3 {
        let color = color;

        let max = color.max_element();
        let mut ratio = color / max;
        let tonemapped_max = Self::tonemap_inner(max, params);

        ratio = ratio.powf(params.saturation / params.cross_saturation);
        ratio = ratio.lerp(Vec3::ONE, tonemapped_max.powf(params.crosstalk));
        ratio = ratio.powf(params.cross_saturation);

        (ratio * tonemapped_max).min(Vec3::ONE).max(Vec3::ZERO)
    }
}

#[derive(Copy, Clone)]
#[repr(C)]
pub struct BakedLottesTonemapperParams {
    a: f32,
    b: f32,
    c: f32,
    d: f32,
    crosstalk: f32,
    saturation: f32,
    cross_saturation: f32,
}

// https://github.com/bevyengine/bevy/blob/2c11ca0291f94b14ee32883d40f8243f3c8e3d6c/pipelined/bevy_hdr/src/bloom.wgsl

// https://github.com/Unity-Technologies/Graphics/blob/master/com.unity.postprocessing/PostProcessing/Shaders/Builtins/Bloom.shader

// http://www.iryoku.com/next-generation-post-processing-in-call-of-duty-advanced-warfare

#[spirv(compute(threads(8, 8, 1)))]
pub fn downsample_pre_filter(
    #[spirv(descriptor_set = 0, binding = 0)] hdr_texture: &Image!(2D, type=f32, sampled),
    #[spirv(descriptor_set = 1, binding = 0)] bloom_texture_mips: &RuntimeArray<
        Image!(2D, format=rgba32f, sampled=false),
    >,
    #[spirv(descriptor_set = 1, binding = 1)] sampler: &Sampler,
    #[spirv(global_invocation_id)] id: IVec3,
) {
    let id = id.truncate();

    let bloom_texture = unsafe { bloom_texture_mips.index(0) };

    let output_size: UVec2 = bloom_texture.query_size();
    let texel_size = Vec2::splat(1.0) / output_size.as_f32();
    let uv = id.as_f32() * texel_size;

    let sample = downsample_box_13_tap(hdr_texture, *sampler, uv, texel_size);

    unsafe {
        bloom_texture.write(id, sample);
    }
}

// . . . . . . .
// . A . B . C .
// . . D . E . .
// . F . G . H .
// . . I . J . .
// . K . L . M .
// . . . . . . .
fn downsample_box_13_tap(
    texture: &Image!(2D, type=f32, sampled),
    sampler: Sampler,
    uv: Vec2,
    texel_size: Vec2,
) -> Vec4 {
    let a: Vec4 = texture.sample_by_lod(sampler, uv + texel_size * Vec2::new(-1.0, -1.0), 0.0);
    let b: Vec4 = texture.sample_by_lod(sampler, uv + texel_size * Vec2::new(0.0, -1.0), 0.0);
    let c: Vec4 = texture.sample_by_lod(sampler, uv + texel_size * Vec2::new(1.0, -1.0), 0.0);
    let d: Vec4 = texture.sample_by_lod(sampler, uv + texel_size * Vec2::new(-0.5, -0.5), 0.0);
    let e: Vec4 = texture.sample_by_lod(sampler, uv + texel_size * Vec2::new(0.5, -0.5), 0.0);
    let f: Vec4 = texture.sample_by_lod(sampler, uv + texel_size * Vec2::new(-1.0, 0.0), 0.0);
    let g: Vec4 = texture.sample_by_lod(sampler, uv, 0.0);
    let h: Vec4 = texture.sample_by_lod(sampler, uv + texel_size * Vec2::new(1.0, 0.0), 0.0);
    let i: Vec4 = texture.sample_by_lod(sampler, uv + texel_size * Vec2::new(-0.5, 0.5), 0.0);
    let j: Vec4 = texture.sample_by_lod(sampler, uv + texel_size * Vec2::new(0.5, 0.5), 0.0);
    let k: Vec4 = texture.sample_by_lod(sampler, uv + texel_size * Vec2::new(-1.0, 1.0), 0.0);
    let l: Vec4 = texture.sample_by_lod(sampler, uv + texel_size * Vec2::new(0.0, 1.0), 0.0);
    let m: Vec4 = texture.sample_by_lod(sampler, uv + texel_size * Vec2::new(1.0, 1.0), 0.0);

    let center_pixels = d + e + i + j;

    let top_left = a + b + f + g;
    let top_right = b + c + g + h;
    let bottom_left = f + g + k + l;
    let bottom_right = g + h + l + m;

    center_pixels * 0.25 * 0.5 + (top_left + top_right + bottom_left + bottom_right) * 0.25 * 0.125
}

//
// Quadratic color thresholding
// curve = (threshold - knee, knee * 2, 0.25 / knee)
//
fn quadratic_threshold(mut color: Vec4, threshold: f32, curve: Vec3) -> Vec4 {
    // Pixel brightness
    let brightness = color.max_element();

    // Under-threshold part: quadratic curve
    let mut rq = clamp(brightness - curve.x, 0.0, curve.y);
    rq = curve.z * rq * rq;

    // Combine and apply the brightness response curve.
    color *= rq.max(brightness - threshold) / brightness.max(core::f32::EPSILON);

    color
}

fn clamp(value: f32, min: f32, max: f32) -> f32 {
    value.max(min).min(max)
}
