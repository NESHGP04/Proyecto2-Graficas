// - Camera (perspective)
// - Animation hooks (camera rotation/dolly) are present but this starter renders a single frame.

use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Instant;
use std::ops::{Add, Sub, Mul, Div, Neg};

use crate::core::{Vec3, Ray};

pub struct Camera { pub origin: Vec3, pub lower_left: Vec3, pub horizontal: Vec3, pub vertical: Vec3 }

impl Camera {
    pub fn new(lookfrom: Vec3, lookat: Vec3, vup: Vec3, vfov: f64, aspect: f64) -> Self {
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
    
    pub fn get_ray(&self, s: f64, t: f64) -> Ray {
        Ray{ orig: self.origin, dir: (self.lower_left + self.horizontal*s + self.vertical*t - self.origin).unit() }
    }
}