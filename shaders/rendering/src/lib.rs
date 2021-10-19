#![cfg_attr(
    target_arch = "spirv",
    feature(register_attr),
    register_attr(spirv),
    no_std
)]

// This needs to be here to provide `#[panic_handler]`.
extern crate spirv_std;

use spirv_std::glam::{Mat4, Vec2, Vec3, Vec4};
use spirv_std::{Image, Sampler};

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
    *output = texture.sample(*sampler, uv);
}
