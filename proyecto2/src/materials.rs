// - Materials: Lambertian (with texture), Metal (reflection), Dielectric (refraction)
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Instant;
use std::ops::{Add, Sub, Mul, Div, Neg};

use crate::core::{Vec3};
use crate::textures::Texture;

#[derive(Clone)]
pub enum Material {
    Lambert { albedo: Vec3, texture: Option<Arc<Texture>> },
    Metal { albedo: Vec3, fuzz: f64 },
    Dielectric { ior: f64 },
}

