#![cfg_attr(
    target_arch = "spirv",
    feature(register_attr),
    register_attr(spirv),
    no_std
)]

// This needs to be here to provide `#[panic_handler]`.
extern crate spirv_std;

use spirv_std::glam::{IVec2, IVec3, Mat4, UVec2, Vec2, Vec3, Vec4};
use spirv_std::{image::SampledImage, Image, RuntimeArray, Sampler};

#[cfg(target_arch = "spirv")]
use spirv_std::num_traits::Float;

#[cfg(not(target_arch = "spirv"))]
use spirv_std::macros::spirv;

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
    *uv = Vec2::new(((vert_idx << 1) & 2) as f32, (vert_idx & 2) as f32);
    let pos = 2.0 * *uv - Vec2::ONE;

    *builtin_pos = Vec4::new(pos.x, pos.y, 0.0, 1.0);
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

// http://www.iryoku.com/next-generation-post-processing-in-call-of-duty-advanced-warfare

// Use a quadratic curve to threshold the colour.
// Taken from:
// https://github.com/Unity-Technologies/Graphics/blob/3c410483bdb6d6f4ced6ba10500b05b6c25ca87f/com.unity.postprocessing/PostProcessing/Shaders/Colors.hlsl#L238-L255
//
// More info:
//
// https://github.com/Unity-Technologies/Graphics/blob/3c410483bdb6d6f4ced6ba10500b05b6c25ca87f/com.unity.postprocessing/PostProcessing/Runtime/Effects/Bloom.cs#L23-L35
fn quadratic_colour_thresholding(color: Vec3, threshold: f32, knee: f32) -> Vec3 {
    fn clamp(value: f32, min: f32, max: f32) -> f32 {
        value.max(min).min(max)
    }

    let curve = Vec3::new(threshold - knee, knee * 2.0, 0.25 / knee);

    let brightness = color.max_element();

    // Under-threshold part: quadratic curve
    let mut rq = clamp(brightness - curve.x, 0.0, curve.y);
    rq = curve.z * rq * rq;

    // Combine and apply the brightness response curve.
    color * rq.max(brightness - threshold) / brightness.max(core::f32::EPSILON)
}

fn calculate_texel_size_and_uv(
    texture: &Image!(2D, format=rgba16f, sampled=false),
    id: IVec2,
) -> (Vec2, Vec2) {
    let output_size: UVec2 = texture.query_size();
    let texel_size = Vec2::splat(1.0) / output_size.as_vec2();
    // Offset the uv by half a texel as we want to sample from the middle, not a corner.
    let uv = (id.as_vec2() + Vec2::splat(0.5)) * texel_size;

    (texel_size, uv)
}

#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct FilterConstants {
    pub threshold: f32,
    pub knee: f32,
}

// Sample and filter the input HDR texture into the 0th half-size mip of the bloom texture.
#[spirv(compute(threads(8, 8)))]
pub fn downsample_initial(
    #[spirv(descriptor_set = 0, binding = 0)] hdr_texture: &SampledImage<
        Image!(2D, type=f32, sampled),
    >,
    #[spirv(descriptor_set = 1, binding = 0)] bloom_texture_mips: &RuntimeArray<
        Image!(2D, format=rgba16f, sampled=false),
    >,
    #[spirv(global_invocation_id)] id: IVec3,
    #[spirv(push_constant)] filter_constants: &FilterConstants,
) {
    let id = id.truncate();

    let bloom_texture = unsafe { bloom_texture_mips.index(0) };

    let (texel_size, uv) = calculate_texel_size_and_uv(bloom_texture, id);

    let sample = sample_13_tap_box_filter(hdr_texture, uv, texel_size, 0);

    // Threshold the colours based on a quadratic curve set by the constants.
    // An alternative to doing this (and what's used in the original presentation)
    // is to use a karis average: http://graphicrants.blogspot.com/2013/12/tone-mapping.html
    //
    // You don't get as much control with that as this colour thresholding though, so I'm using
    // this instead.
    let thresholded =
        quadratic_colour_thresholding(sample, filter_constants.threshold, filter_constants.knee);

    unsafe {
        bloom_texture.write(id, thresholded.extend(1.0));
    }
}

// Sample the bloom texture at mip N and write to mip N + 1.
#[spirv(compute(threads(8, 8)))]
pub fn downsample(
    #[spirv(descriptor_set = 0, binding = 0)] source_texture: &SampledImage<
        Image!(2D, type=f32, sampled),
    >,
    #[spirv(descriptor_set = 1, binding = 0)] destination_textures: &RuntimeArray<
        Image!(2D, format=rgba16f, sampled=false),
    >,
    #[spirv(global_invocation_id)] id: IVec3,
    #[spirv(push_constant)] source_mip: &u32,
) {
    let id = id.truncate();

    let destination_texture = unsafe { destination_textures.index((*source_mip + 1) as usize) };

    let (texel_size, uv) = calculate_texel_size_and_uv(destination_texture, id);

    let sample = sample_13_tap_box_filter(source_texture, uv, texel_size, *source_mip).extend(1.0);

    unsafe {
        destination_texture.write(id, sample);
    }
}

// Sample the bloom texture at mip N + 1, perform additive blending with the texture at mip N and write to mip N.
#[spirv(compute(threads(8, 8)))]
pub fn upsample(
    #[spirv(descriptor_set = 0, binding = 0)] source_texture: &SampledImage<
        Image!(2D, type=f32, sampled),
    >,
    #[spirv(descriptor_set = 1, binding = 0)] destination_textures: &RuntimeArray<
        Image!(2D, format=rgba16f, sampled=false),
    >,
    #[spirv(global_invocation_id)] id: IVec3,
    #[spirv(push_constant)] dest_mip: &u32,
) {
    let id = id.truncate();

    let destination_texture = unsafe { destination_textures.index(*dest_mip as usize) };

    let (texel_size, uv) = calculate_texel_size_and_uv(destination_texture, id);

    let sample = sample_3x3_tent_filter(source_texture, uv, texel_size, dest_mip + 1).extend(1.0);

    let existing_sample: Vec4 = destination_texture.read(id);

    unsafe {
        destination_texture.write(id, existing_sample + sample);
    }
}

// Sample the bloom texture for a final time at the 0th mip and perform additive blending with the hdr texture.
#[spirv(compute(threads(8, 8)))]
pub fn upsample_final(
    #[spirv(descriptor_set = 0, binding = 0)] source_texture: &SampledImage<
        Image!(2D, type=f32, sampled),
    >,
    #[spirv(descriptor_set = 1, binding = 0)] hdr_texture: &Image!(2D, format=rgba16f, sampled=false),
    #[spirv(global_invocation_id)] id: IVec3,
) {
    let id = id.truncate();

    let (texel_size, uv) = calculate_texel_size_and_uv(hdr_texture, id);

    let sample = sample_3x3_tent_filter(source_texture, uv, texel_size, 0).extend(1.0);

    let existing_sample: Vec4 = hdr_texture.read(id);

    unsafe {
        hdr_texture.write(id, existing_sample + sample);
    }
}

trait CombinedTextureSampler {
    fn sample(&self, uv: Vec2, lod: f32) -> Vec4;

    fn sample_vec3_by_lod(&self, uv: Vec2, lod: u32) -> Vec3 {
        self.sample(uv, lod as f32).truncate()
    }
}

impl CombinedTextureSampler for (&Image!(2D, type = f32, sampled), &Sampler) {
    fn sample(&self, uv: Vec2, lod: f32) -> Vec4 {
        self.0.sample_by_lod(*self.1, uv, lod)
    }
}

impl CombinedTextureSampler for &SampledImage<Image!(2D, type = f32, sampled)> {
    fn sample(&self, uv: Vec2, lod: f32) -> Vec4 {
        unsafe { self.sample_by_lod(uv, lod) }
    }
}

// Take 13 samples in a grid around the center pixel:
// . . . . . . .
// . A . B . C .
// . . D . E . .
// . F . G . H .
// . . I . J . .
// . K . L . M .
// . . . . . . .
// These samples are interpreted as 4 overlapping boxes
// plus a center box.
#[rustfmt::skip]
#[allow(clippy::many_single_char_names)]
fn sample_13_tap_box_filter<T: CombinedTextureSampler>(
    texture: T,
    uv: Vec2,
    texel_size: Vec2,
    lod: u32,
) -> Vec3 {
    let a = texture.sample_vec3_by_lod(uv + texel_size * Vec2::new(-1.0, -1.0), lod);
    let b = texture.sample_vec3_by_lod(uv + texel_size * Vec2::new(0.0, -1.0),  lod);
    let c = texture.sample_vec3_by_lod(uv + texel_size * Vec2::new(1.0, -1.0),  lod);
    let d = texture.sample_vec3_by_lod(uv + texel_size * Vec2::new(-0.5, -0.5), lod);
    let e = texture.sample_vec3_by_lod(uv + texel_size * Vec2::new(0.5, -0.5),  lod);
    let f = texture.sample_vec3_by_lod(uv + texel_size * Vec2::new(-1.0, 0.0),  lod);
    let g = texture.sample_vec3_by_lod(uv, lod);
    let h = texture.sample_vec3_by_lod(uv + texel_size * Vec2::new(1.0, 0.0),  lod);
    let i = texture.sample_vec3_by_lod(uv + texel_size * Vec2::new(-0.5, 0.5), lod);
    let j = texture.sample_vec3_by_lod(uv + texel_size * Vec2::new(0.5, 0.5),  lod);
    let k = texture.sample_vec3_by_lod(uv + texel_size * Vec2::new(-1.0, 1.0), lod);
    let l = texture.sample_vec3_by_lod(uv + texel_size * Vec2::new(0.0, 1.0),  lod);
    let m = texture.sample_vec3_by_lod(uv + texel_size * Vec2::new(1.0, 1.0),  lod);

    let center_pixels = d + e + i + j;

    let top_left = a + b + f + g;
    let top_right = b + c + g + h;
    let bottom_left = f + g + k + l;
    let bottom_right = g + h + l + m;

    // The center box (after having been averaged by dividing by 4) has a weight of 0.5
    // and the other boxes have a weight of 0.125.
    center_pixels * 0.25 * 0.5 + (top_left + top_right + bottom_left + bottom_right) * 0.25 * 0.125
}

// Sample in a 3x3 grid but with weights to produce a tent filter:
//
//        a*1 b*2 c*1
// 1/16 * d*2 e*4 f*2
//        g*1 h*2 i*1
#[rustfmt::skip]
#[allow(clippy::many_single_char_names)]
fn sample_3x3_tent_filter<T: CombinedTextureSampler>(
    texture: T,
    uv: Vec2,
    texel_size: Vec2,
    lod: u32
) -> Vec3 {
    let a = texture.sample_vec3_by_lod(uv + texel_size * Vec2::new(-1.0, -1.0), lod);
    let b = texture.sample_vec3_by_lod(uv + texel_size * Vec2::new(0.0, -1.0),  lod);
    let c = texture.sample_vec3_by_lod(uv + texel_size * Vec2::new(1.0, -1.0),  lod);
    let d = texture.sample_vec3_by_lod(uv + texel_size * Vec2::new(-1.0, 0.0),  lod);
    let e = texture.sample_vec3_by_lod(uv, lod);
    let f = texture.sample_vec3_by_lod(uv + texel_size * Vec2::new(1.0, 0.0),  lod);
    let g = texture.sample_vec3_by_lod(uv + texel_size * Vec2::new(-1.0, 1.0), lod);
    let h = texture.sample_vec3_by_lod(uv + texel_size * Vec2::new(0.0, 1.0),  lod);
    let i = texture.sample_vec3_by_lod(uv + texel_size * Vec2::new(1.0, 1.0),  lod);

    ((a + c + g + i) + (b + d + f + h) * 2.0 + e * 4.0) / 16.0
}
