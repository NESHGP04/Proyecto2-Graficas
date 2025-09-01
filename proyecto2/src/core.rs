//Geometría y operaciones básicas
use std::ops::{Add, Sub, Mul, Div, Neg};

#[derive(Clone, Copy, Debug)]
pub struct Vec3 {pub x: f64, pub y: f64, pub z: f64 }

impl Vec3 {
    pub fn new(x: f64, y: f64, z: f64) -> Self { Self { x, y, z } }
    pub fn zero() -> Self { Self::new(0.0,0.0,0.0) }
    pub fn length(&self) -> f64 { (self.x*self.x + self.y*self.y + self.z*self.z).sqrt() }
    
    pub fn len2(&self) -> f64 {
        self.x * self.x + self.y * self.y + self.z * self.z
    }

    pub fn len(&self) -> f64 {
        self.len2().sqrt()
    }

    pub fn unit(&self) -> Self { let k = 1.0/self.length(); Self::new(self.x*k, self.y*k, self.z*k) }

    pub fn dot(a: Self, b: Self) -> f64 { a.x*b.x + a.y*b.y + a.z*b.z }
    
    pub fn cross(a: Self, b: Self) -> Self {
        Self::new(
            a.y*b.z - a.z*b.y,
            a.z*b.x - a.x*b.z,
            a.x*b.y - a.y*b.x,
        )
    }
}

// Operadores
impl Add for Vec3 { type Output = Self; fn add(self, o: Self) -> Self { Self::new(self.x+o.x,self.y+o.y,self.z+o.z) } }
impl Sub for Vec3 { type Output = Self; fn sub(self, o: Self) -> Self { Self::new(self.x-o.x,self.y-o.y,self.z-o.z) } }
impl Mul for Vec3 {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        Self::new(self.x * rhs.x, self.y * rhs.y, self.z * rhs.z)
    }
}
impl Mul<f64> for Vec3 { type Output = Self; fn mul(self, s: f64) -> Self { Self::new(self.x*s,self.y*s,self.z*s) } }
impl Div<f64> for Vec3 { type Output = Self; fn div(self, s: f64) -> Self { Self::new(self.x/s,self.y/s,self.z/s) } }
impl Neg for Vec3 { type Output = Self; fn neg(self) -> Self { Self::new(-self.x,-self.y,-self.z) } }

// ---------------------------- Ray & Hit ----------------------------
#[derive(Clone, Copy, Debug)]
pub struct Ray { pub orig: Vec3, pub dir: Vec3 }
impl Ray { pub fn at(&self,t:f64)->Vec3 { self.orig + self.dir * t } }

#[derive(Clone, Copy, Debug)]
pub struct HitRecord {
    pub p: Vec3,
    pub normal: Vec3,
    pub t: f64,
    pub u: f64,
    pub v: f64,
    pub front_face: bool,
    pub mat_id: usize,
}

pub fn set_face_normal(ray: &Ray, outward_normal: Vec3) -> (Vec3,bool) {
    let front = Vec3::dot(ray.dir, outward_normal) < 0.0;
    ( if front { outward_normal } else { -outward_normal }, front )
}

// ---------------------------- Shading / scattering ----------------------------
pub fn reflect(v: Vec3, n: Vec3) -> Vec3 { v - n * (2.0 * Vec3::dot(v,n)) }

pub fn refract(uv: Vec3, n: Vec3, etai_over_etat: f64) -> Vec3 {
    let cos_theta = Vec3::dot(-uv, n).min(1.0);
    let r_out_perp = (uv + n * cos_theta) * etai_over_etat;
    let r_out_parallel = n * -((1.0 - r_out_perp.len2()).abs().sqrt());
    r_out_perp + r_out_parallel
}

pub fn schlick(cosine: f64, ref_idx: f64) -> f64 {
    let r0 = ((1.0 - ref_idx)/(1.0 + ref_idx)).powi(2);
    r0 + (1.0 - r0) * (1.0 - cosine).powi(5)
}
