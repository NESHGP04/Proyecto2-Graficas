// raytracer_cozy_cafe.rs
// Single-file CPU path tracer in Rust (no external crates)
// Produces a PPM image. Textures must be PPM (P3) files placed in `assets/textures/`.
// Features implemented:
// - Camera (perspective)
// - Axis-aligned cubes (AABB intersection + face-based UV mapping)
// - Materials: Lambertian (with texture), Metal (reflection), Dielectric (refraction)
// - Skybox (simple gradient)
// - Multithreaded tile renderer using std::thread and available_parallelism
// - Simple BVH-like spatial grouping (optional naive split)
// - Animation hooks (camera rotation/dolly) are present but this starter renders a single frame.

use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Instant;
use std::ops::{Add, Sub, Mul, Div, Neg};

mod bvh;
mod camera;
mod materials;
mod render;
mod scene;
mod textures;
mod shapes;
mod core;

use crate::camera::Camera;
use crate::materials::Material;
use crate::scene::{Scene, make_scene};
use crate::render::{render_image, color_at};
use crate::shapes::cube::Cube;
use crate::textures::Texture;

fn main() {
    let width = 800;
    let height = 450;
    let samples = 40;
    let max_depth = 6;

    let scene = Arc::new(make_scene());
    let lookfrom = crate::core::Vec3::new(0.0, 0.3, 4.0);
    let lookat = crate::core::Vec3::new(0.0, -0.2, 0.0);
    let cam = Arc::new(Camera::new(lookfrom, lookat, crate::core::Vec3::new(0.0,1.0,0.0), 40.0, width as f64 / height as f64));

    render_image(width, height, samples, max_depth, scene, cam, "outputs/cozy_cafe.ppm");
}