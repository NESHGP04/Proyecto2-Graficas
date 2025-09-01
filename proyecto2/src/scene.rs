//objetos, transforms
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Instant;
use std::ops::{Add, Sub, Mul, Div, Neg};

use crate::core::{Vec3, HitRecord, Ray, set_face_normal};
use crate::Cube;
use crate::materials::Material;
use crate::textures::Texture;

pub struct Scene { pub cubes: Vec<Cube>, pub materials: Vec<Material>, pub sky_color_top: Vec3, pub sky_color_bottom: Vec3, pub textures: Vec<Option<Arc<Texture>>> }

impl Scene {
    pub fn new() -> Self {
        Self {
            cubes: Vec::new(),
            materials: Vec::new(),
            sky_color_top: Vec3::new(0.5, 0.7, 1.0),
            sky_color_bottom: Vec3::new(1.0, 1.0, 1.0),
            textures: Vec::new(),
        }
    }
    
    pub fn hit(&self, r: &Ray, t_min: f64, t_max: f64) -> Option<HitRecord> {
        let mut closest = t_max; let mut rec: Option<HitRecord> = None;
        for c in &self.cubes {
            if let Some(hr) = c.intersects(r, t_min, closest) {
                closest = hr.t; rec = Some(hr);
            }
        }
        rec
    }
}

pub fn make_scene() -> Scene {
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