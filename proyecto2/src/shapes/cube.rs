use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Instant;
use std::ops::{Add, Sub, Mul, Div, Neg};

use crate::core::{Vec3, Ray, set_face_normal, HitRecord};

// ---------------------------- Objects: Axis-aligned cube ----------------------------
pub struct Cube { pub min: Vec3, pub max: Vec3, pub mat_id: usize }

impl Cube {
    pub fn intersects(&self, r: &Ray, t_min: f64, t_max: f64) -> Option<HitRecord> {
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