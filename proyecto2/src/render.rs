// - Multithreaded tile renderer using std::thread and available_parallelism
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Instant;
use std::ops::{Add, Sub, Mul, Div, Neg};

use crate::core::{Vec3, Ray, reflect, refract, schlick};
use crate::scene::Scene;
use crate::camera::Camera;
use crate::materials::Material;
use crate::shapes::cube;

pub struct Rng { pub state: u64, }

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    pub fn next_u64(&mut self) -> u64 {
        // Xorshift64*
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    pub fn next_f64(&mut self) -> f64 {
        // Uniform en [0,1)
        (self.next_u64() as f64) / (u64::MAX as f64 + 1.0)
    }

    pub fn unit(&mut self) -> Vec3 {
        // Vector unitario aleatorio en la esfera
        let a = 2.0 * std::f64::consts::PI * self.next_f64();
        let z = self.next_f64() * 2.0 - 1.0;
        let r = (1.0 - z * z).sqrt();
        Vec3::new(r * a.cos(), r * a.sin(), z)
    }

    pub fn in_unit_sphere(&mut self) -> Vec3 {
        loop {
            let p = Vec3::new(
                self.next_f64() * 2.0 - 1.0,
                self.next_f64() * 2.0 - 1.0,
                self.next_f64() * 2.0 - 1.0,
            );
            if p.len2() < 1.0 {
                return p;
            }
        }
    }
}

pub fn color_at(ray: &Ray, scene: &Scene, depth: usize, rng: &mut Rng) -> Vec3 {
    if depth == 0 { return Vec3::zero(); }
    if let Some(hit) = scene.hit(ray, 0.001, 1e9) {
        let mat = &scene.materials[hit.mat_id];
        match mat {
            // Lambertian / difuso
            Material::Lambert { albedo, texture } => {
                // Obtener el color de la textura si existe
                let mut tex_col = *albedo;
                if let Some(t) = texture {
                    tex_col = t.sample(hit.u, hit.v);
                }

                // Ray hacia un punto aleatorio dentro de la esfera unidad alrededor de la normal
                let target = hit.p + hit.normal + rng.in_unit_sphere();
                let scattered_ray = Ray {
                    orig: hit.p,
                    dir: (target - hit.p).unit(),
                };

                // Retornar el color del raytracing multiplicado por el albedo
                return tex_col * color_at(&scattered_ray, scene, depth-1, rng);
            }

            // Metal / reflectivo
            Material::Metal { albedo, fuzz } => {
                let reflected = reflect(ray.dir.unit(), hit.normal);
                let scattered = reflected + rng.in_unit_sphere() * (*fuzz);

                if Vec3::dot(scattered, hit.normal) > 0.0 {
                    let scattered_ray = Ray {
                        orig: hit.p,
                        dir: scattered.unit(),
                    };
                    return *albedo * color_at(&scattered_ray, scene, depth-1, rng);
                } else {
                    return Vec3::zero();
                }
            }

            // Dielectric / transparente/refractiva
            Material::Dielectric { ior } => {
                let ref_ratio = if hit.front_face { 1.0 / *ior } else { *ior };
                let unit_dir = ray.dir.unit();
                let cos_theta = Vec3::dot(-unit_dir, hit.normal).min(1.0);
                let sin_theta = (1.0 - cos_theta * cos_theta).sqrt();
                let cannot_refract = ref_ratio * sin_theta > 1.0;

                let direction = if cannot_refract || schlick(cos_theta, *ior) > rng.next_f64() {
                    reflect(unit_dir, hit.normal)
                } else {
                    refract(unit_dir, hit.normal, ref_ratio)
                };

                let scattered_ray = Ray {
                    orig: hit.p,
                    dir: direction.unit(),
                };

                return color_at(&scattered_ray, scene, depth-1, rng);
            }

            //Luz interior
            Material::Emissive { color } => {
                return *color;
            }
        }

    }
    // skybox gradient
    let t = 0.5 * (ray.dir.unit().y + 1.0);
    scene.sky_color_bottom*(1.0 - t) + scene.sky_color_top * t
}

pub fn render_image(width: usize, height: usize, samples: usize, max_depth: usize, scene: Arc<Scene>, cam: Arc<Camera>, out_path: &str){
    let start = Instant::now();
    let num_threads = thread::available_parallelism().map(|n| n.get()).unwrap_or(4) as usize;
    let tile_h = (height + num_threads - 1) / num_threads;
    let pixels = Arc::new(Mutex::new(vec![Vec3::zero(); width*height]));
    let mut handles = Vec::new();
    for tid in 0..num_threads {
        let scene_cl = scene.clone();
        let pixels_cl = Arc::clone(&pixels);
        let cam_cl = cam.clone();
        let hstart = tid * tile_h;
        let hend = ((tid+1)*tile_h).min(height);
        let w = width; let h = height; let s = samples; let md = max_depth;
        let handle = thread::spawn(move || {
            let mut rng = Rng::new((tid as u64 + 1) * 0x9E3779B97F4A7C15u64);
            for j in hstart..hend {
                for i in 0..w {
                    let mut col = Vec3::zero();
                    for _ in 0..s {
                        let u = (i as f64 + rng.next_f64()) / (w as f64 - 1.0);
                        let v = ((h-1-j) as f64 + rng.next_f64()) / (h as f64 - 1.0);
                        let r = cam_cl.get_ray(u, v);
                        col = col + color_at(&r, &scene_cl, md, &mut rng);
                    }
                    col = col / (s as f64);
                    // simple gamma correction
                    col = Vec3::new(col.x.sqrt(), col.y.sqrt(), col.z.sqrt());
                    let mut px = pixels_cl.lock().unwrap();
                    px[j*w + i] = col;
                }
            }
        });
        handles.push(handle);
    }
    for h in handles { h.join().unwrap(); }
    let pixels = Arc::try_unwrap(pixels).unwrap().into_inner().unwrap();
    // write PPM
    let mut file = File::create(out_path).expect("cannot create output");
    write!(file, "P3\n{} {}\n255\n", width, height).unwrap();
    for p in pixels {
        let ir = (255.999 * p.x.clamp(0.0,1.0)) as u32;
        let ig = (255.999 * p.y.clamp(0.0,1.0)) as u32;
        let ib = (255.999 * p.z.clamp(0.0,1.0)) as u32;
        write!(file, "{} {} {}\n", ir, ig, ib).unwrap();
    }
    println!("Rendered {}x{} in {:.2}s -> {}", width, height, start.elapsed().as_secs_f64(), out_path);
}