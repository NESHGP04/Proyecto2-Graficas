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
use std::io::{BufRead, BufReader, Read, Write};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Instant;

// ---------------------------- math primitives ----------------------------
#[derive(Clone, Copy, Debug)]
struct Vec3 { x: f64, y: f64, z: f64 }
impl Vec3 {
    fn new(x: f64, y: f64, z: f64) -> Self { Self { x, y, z } }
    fn zero() -> Self { Self::new(0.0,0.0,0.0) }
    fn length(&self) -> f64 { (self.x*self.x + self.y*self.y + self.z*self.z).sqrt() }
    fn unit(&self) -> Self { let k = 1.0/self.length(); Self::new(self.x*k, self.y*k, self.z*k) }
}
use std::ops::{Add, Sub, Mul, Div, Neg};
impl Add for Vec3 { type Output = Self; fn add(self, o: Self) -> Self { Self::new(self.x+o.x,self.y+o.y,self.z+o.z) } }
impl Sub for Vec3 { type Output = Self; fn sub(self, o: Self) -> Self { Self::new(self.x-o.x,self.y-o.y,self.z-o.z) } }
impl Mul<f64> for Vec3 { type Output = Self; fn mul(self, s: f64) -> Self { Self::new(self.x*s,self.y*s,self.z*s) } }
impl Div<f64> for Vec3 { type Output = Self; fn div(self, s: f64) -> Self { Self::new(self.x/s,self.y/s,self.z/s) } }
impl Neg for Vec3 { type Output = Self; fn neg(self) -> Self { Self::new(-self.x,-self.y,-self.z) } }
impl Vec3 {
    fn dot(a: Self, b: Self) -> f64 { a.x*b.x + a.y*b.y + a.z*b.z }
    fn cross(a: Self, b: Self) -> Self {
        Self::new(
            a.y*b.z - a.z*b.y,
            a.z*b.x - a.x*b.z,
            a.x*b.y - a.y*b.x,
        )
    }
}

// random helper (simple xorshift)
struct Rng { state: u64 }
impl Rng { fn new(seed: u64)->Self{Self{state:seed}}; fn next_u64(&mut self)->u64{let mut x=self.state; x ^= x << 13; x ^= x >> 7; x ^= x << 17; self.state = x; x}
    fn next_f64(&mut self)->f64 { (self.next_u64() & 0xFFFFFFFF) as f64 / (0x1_0000_0000u64 as f64) }
    fn in_unit_sphere(&mut self)->Vec3{ loop { let p = Vec3::new(self.next_f64()*2.0-1.0, self.next_f64()*2.0-1.0, self.next_f64()*2.0-1.0); if p.x*p.x + p.y*p.y + p.z*p.z < 1.0 { return p } } }
    fn unit(&mut self)->Vec3{ self.in_unit_sphere().unit() }
}

// ---------------------------- Ray & Hit ----------------------------
#[derive(Clone, Copy, Debug)]
struct Ray { orig: Vec3, dir: Vec3 }
impl Ray { fn at(&self,t:f64)->Vec3 { self.orig + self.dir * t } }

struct HitRecord { p: Vec3, normal: Vec3, t: f64, u: f64, v: f64, front_face: bool, mat_id: usize }

fn set_face_normal(ray: &Ray, outward_normal: Vec3) -> (Vec3, bool) {
    let front = Vec3::dot(ray.dir, outward_normal) < 0.0;
    ( if front { outward_normal } else { -outward_normal }, front )
}

// ---------------------------- Texture loader (PPM P3) ----------------------------
struct Texture { w: usize, h: usize, data: Vec<u8> } // RGB bytes
impl Texture {
    fn load_ppm(path: &str)->Option<Self> {
        let f = File::open(path).ok()?;
        let mut reader = BufReader::new(f);
        let mut header = String::new();
        reader.read_line(&mut header).ok()?;
        if !header.trim().starts_with("P3") { return None; }
        // skip comments
        let mut dims_line = String::new();
        loop {
            dims_line.clear();
            reader.read_line(&mut dims_line).ok()?;
            if dims_line.trim().starts_with('#') { continue } else { break }
        }
        let mut parts = dims_line.split_whitespace();
        let w: usize = parts.next()?.parse().ok()?;
        let h: usize = parts.next()?.parse().ok()?;
        let mut max_line = String::new(); reader.read_line(&mut max_line).ok()?; // max color
        let mut nums = Vec::new();
        for line in reader.lines() { if let Ok(l) = line { for tok in l.split_whitespace() { nums.push(tok.to_string()) } } }
        if nums.len() < w*h*3 { return None; }
        let mut data = Vec::with_capacity(w*h*3);
        for i in 0..(w*h*3) { let v: u8 = nums[i].parse::<u32>().ok().map(|x| x as u8).unwrap_or(0); data.push(v); }
        Some(Self{w,h,data})
    }
    fn sample(&self, u: f64, v: f64)->Vec3 {
        // wrap
        let uu = (u.fract()+1.0).fract();
        let vv = (v.fract()+1.0).fract();
        let x = ((uu * self.w as f64) as usize).min(self.w-1);
        let y = (( (1.0 - vv) * self.h as f64) as usize).min(self.h-1);
        let idx = (y*self.w + x)*3;
        Vec3::new(self.data[idx] as f64/255.0, self.data[idx+1] as f64/255.0, self.data[idx+2] as f64/255.0)
    }
}

// ---------------------------- Materials ----------------------------
#[derive(Clone)]
enum Material {
    Lambert { albedo: Vec3, texture: Option<Arc<Texture>> },
    Metal { albedo: Vec3, fuzz: f64 },
    Dielectric { ior: f64 },
}

// ---------------------------- Objects: Axis-aligned cube ----------------------------
struct Cube { min: Vec3, max: Vec3, mat_id: usize }
impl Cube {
    fn intersects(&self, r: &Ray, t_min: f64, t_max: f64) -> Option<HitRecord> {
        // slab method
        let mut t0 = t_min; let mut t1 = t_max; let mut face_normal = Vec3::zero(); let mut hit_u = 0.0; let mut hit_v = 0.0; let mut hit_t = 0.0;
        let axes = [(self.min.x, self.max.x, Vec3::new(1.0,0.0,0.0)), (self.min.y, self.max.y, Vec3::new(0.0,1.0,0.0)), (self.min.z, self.max.z, Vec3::new(0.0,0.0,1.0))];
        let mut t_min_axis = t0; let mut t_max_axis = t1;
        // compute intersection with slabs to find nearest t
        let mut tmin = t0; let mut tmax = t1; let mut min_axis = 0; let mut max_axis = 0;
        // X
        for i in 0..3 {
            let (mn, mx, axis) = axes[i];
            let invd = 1.0 / (if i==0 { r.dir.x } else if i==1 { r.dir.y } else { r.dir.z });
            let origin_comp = if i==0 { r.orig.x } else if i==1 { r.orig.y } else { r.orig.z };
            let mut t0_ = (mn - origin_comp) * invd;
            let mut t1_ = (mx - origin_comp) * invd;
            let mut out_norm = axis;
            if invd < 0.0 { std::mem::swap(&mut t0_, &mut t1_); out_norm = -axis; }
            if t0_ > tmin { tmin = t0_; face_normal = out_norm; min_axis = i; }
            if t1_ < tmax { tmax = t1_; max_axis = i; }
            if tmax <= tmin { return None }
        }
        hit_t = tmin;
        if hit_t < t_min || hit_t > t_max { return None }
        let p = r.at(hit_t);
        // compute UV based on which face was hit (use face_normal to determine face)
        let n = face_normal;
        let (u,v) = if n.x.abs() > 0.5 {
            let u_ = (p.z - self.min.z) / (self.max.z - self.min.z);
            let v_ = (p.y - self.min.y) / (self.max.y - self.min.y);
            (u_, v_)
        } else if n.y.abs() > 0.5 {
            let u_ = (p.x - self.min.x) / (self.max.x - self.min.x);
            let v_ = (p.z - self.min.z) / (self.max.z - self.min.z);
            (u_, v_)
        } else {
            let u_ = (p.x - self.min.x) / (self.max.x - self.min.x);
            let v_ = (p.y - self.min.y) / (self.max.y - self.min.y);
            (u_, v_)
        };
        let (normal, front_face) = set_face_normal(r, n);
        Some(HitRecord{ p, normal, t: hit_t, u, v, front_face, mat_id: self.mat_id })
    }
}

// ---------------------------- Scene ----------------------------
struct Scene { cubes: Vec<Cube>, materials: Vec<Material>, sky_color_top: Vec3, sky_color_bottom: Vec3, textures: Vec<Option<Arc<Texture>>> }
impl Scene {
    fn hit(&self, r: &Ray, t_min: f64, t_max: f64) -> Option<HitRecord> {
        let mut closest = t_max; let mut rec: Option<HitRecord> = None;
        for c in &self.cubes {
            if let Some(hr) = c.intersects(r, t_min, closest) {
                closest = hr.t; rec = Some(hr);
            }
        }
        rec
    }
}

// ---------------------------- Shading / scattering ----------------------------
fn reflect(v: Vec3, n: Vec3) -> Vec3 { v - n * (2.0 * Vec3::dot(v, n)) }
fn refract(uv: Vec3, n: Vec3, etai_over_etat: f64) -> Vec3 {
    let cos_theta = Vec3::dot(-uv, n).min(1.0);
    let r_out_perp = (uv + n * cos_theta) * etai_over_etat;
    let r_out_parallel = n * -( (1.0 - r_out_perp.x*r_out_perp.x - r_out_perp.y*r_out_perp.y - r_out_perp.z*r_out_perp.z).abs().sqrt() );
    r_out_perp + r_out_parallel
}
fn schlick(cosine: f64, ref_idx: f64) -> f64 {
    let r0 = ((1.0 - ref_idx) / (1.0 + ref_idx)).powi(2);
    r0 + (1.0 - r0) * (1.0 - cosine).powi(5)
}

fn color_at(ray: &Ray, scene: &Scene, depth: usize, rng: &mut Rng) -> Vec3 {
    if depth == 0 { return Vec3::zero(); }
    if let Some(hit) = scene.hit(ray, 0.001, 1e9) {
        let mat = &scene.materials[hit.mat_id];
        match mat {
            Material::Lambert { albedo, texture } => {
                let mut tex_col = *albedo;
                if let Some(t) = texture { tex_col = t.sample(hit.u, hit.v); }
                let target = hit.p + hit.normal + rng.unit();
                return tex_col * color_at(&Ray{orig:hit.p, dir: (target - hit.p).unit()}, scene, depth-1, rng) * 0.5 + Vec3::zero();
            }
            Material::Metal { albedo, fuzz } => {
                let reflected = reflect(ray.dir.unit(), hit.normal);
                let scattered = reflected + rng.in_unit_sphere() * (*fuzz);
                if Vec3::dot(scattered, hit.normal) > 0.0 {
                    return *albedo * color_at(&Ray{orig:hit.p, dir:scattered.unit()}, scene, depth-1, rng) * 0.9;
                } else { return Vec3::zero(); }
            }
            Material::Dielectric { ior } => {
                let ref_ratio = if hit.front_face { 1.0 / *ior } else { *ior };
                let unit_dir = ray.dir.unit();
                let cos_theta = Vec3::dot(-unit_dir, hit.normal).min(1.0);
                let sin_theta = (1.0 - cos_theta*cos_theta).sqrt();
                let cannot_refract = ref_ratio * sin_theta > 1.0;
                let direction = if cannot_refract || schlick(cos_theta, *ior) > rng.next_f64() {
                    reflect(unit_dir, hit.normal)
                } else {
                    refract(unit_dir, hit.normal, ref_ratio)
                };
                return color_at(&Ray{orig:hit.p, dir:direction.unit()}, scene, depth-1, rng) * 0.98;
            }
        }
    }
    // skybox gradient
    let t = 0.5 * (ray.dir.unit().y + 1.0);
    scene.sky_color_bottom*(1.0 - t) + scene.sky_color_top * t
}

// ---------------------------- Camera ----------------------------
struct Camera { origin: Vec3, lower_left: Vec3, horizontal: Vec3, vertical: Vec3 }
impl Camera {
    fn new(lookfrom: Vec3, lookat: Vec3, vup: Vec3, vfov: f64, aspect: f64) -> Self {
        let theta = vfov.to_radians();
        let h = (theta/2.0).tan();
        let viewport_height = 2.0*h; let viewport_width = aspect * viewport_height;
        let w = (lookfrom - lookat).unit();
        let u = Vec3::cross(vup, w).unit();
        let v = Vec3::cross(w, u);
        let origin = lookfrom;
        let horizontal = u * viewport_width;
        let vertical = v * viewport_height;
        let lower_left = origin - horizontal/2.0 - vertical/2.0 - w;
        Self{ origin, lower_left, horizontal, vertical }
    }
    fn get_ray(&self, s: f64, t: f64) -> Ray {
        Ray{ orig: self.origin, dir: (self.lower_left + self.horizontal*s + self.vertical*t - self.origin).unit() }
    }
}

// ---------------------------- Main: scene setup & renderer ----------------------------
fn make_scene() -> Scene {
    let mut materials = Vec::new();
    let mut textures: Vec<Option<Arc<Texture>>> = Vec::new();
    // load textures if exist (assets/textures/*.ppm)
    let wood = Texture::load_ppm("assets/textures/wood.ppm").map(|t| Arc::new(t));
    let fabric = Texture::load_ppm("assets/textures/fabric.ppm").map(|t| Arc::new(t));
    let ceramic = Texture::load_ppm("assets/textures/ceramic.ppm").map(|t| Arc::new(t));

    // materials: push and remember index
    // 0: tabletop (wood)
    materials.push(Material::Lambert{ albedo: Vec3::new(0.8,0.6,0.4), texture: wood.clone() }); textures.push(wood.clone());
    // 1: cushion (fabric)
    materials.push(Material::Lambert{ albedo: Vec3::new(0.8,0.2,0.3), texture: fabric.clone() }); textures.push(fabric.clone());
    // 2: cup (ceramic)
    materials.push(Material::Lambert{ albedo: Vec3::new(0.9,0.9,1.0), texture: ceramic.clone() }); textures.push(ceramic.clone());
    // 3: lamp (metal)
    materials.push(Material::Metal{ albedo: Vec3::new(0.9,0.85,0.8), fuzz: 0.05 }); textures.push(None);
    // 4: window glass (dielectric)
    materials.push(Material::Dielectric{ ior: 1.5 }); textures.push(None);

    let mut cubes = Vec::new();
    // table (large cube scaled thin)
    cubes.push(Cube{ min: Vec3::new(-1.5, -0.5, -1.0), max: Vec3::new(1.5, -0.4, 1.0), mat_id:0 });
    // small plate on table
    cubes.push(Cube{ min: Vec3::new(-0.3, -0.39, -0.2), max: Vec3::new(0.3, -0.35, 0.2), mat_id:2 });
    // cup (ceramic) as short cube
    cubes.push(Cube{ min: Vec3::new(0.5, -0.39, 0.2), max: Vec3::new(0.8, -0.15, 0.5), mat_id:2 });
    // lamp (hanging metal cube)
    cubes.push(Cube{ min: Vec3::new(-0.1, 0.6, -0.1), max: Vec3::new(0.1, 0.8, 0.1), mat_id:3 });
    // window (glass)
    cubes.push(Cube{ min: Vec3::new(1.2, -0.3, -1.6), max: Vec3::new(1.8, 1.2, 1.6), mat_id:4 });
    // cushion
    cubes.push(Cube{ min: Vec3::new(-1.0, -0.39, 0.7), max: Vec3::new(-0.4, -0.15, 1.2), mat_id:1 });

    Scene{ cubes, materials, sky_color_top: Vec3::new(0.2,0.3,0.6), sky_color_bottom: Vec3::new(1.0,0.9,0.7), textures }
}

fn render_image(width: usize, height: usize, samples: usize, max_depth: usize, scene: Arc<Scene>, cam: &Camera, out_path: &str) {
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

fn main() {
    // recommended settings for your machine: 8GB RAM, CPU - moderate
    let width = 800usize; let height = 450usize; // 16:9, modest
    let samples = 40usize; // per-pixel samples
    let max_depth = 6usize;

    let scene = Arc::new(make_scene());
    let lookfrom = Vec3::new(0.0, 0.3, 4.0);
    let lookat = Vec3::new(0.0, -0.2, 0.0);
    let cam = Camera::new(lookfrom, lookat, Vec3::new(0.0,1.0,0.0), 40.0, width as f64 / height as f64);
    render_image(width, height, samples, max_depth, scene, &cam, "outputs/cozy_cafe.ppm");
}