use crate::vec3::Vector3;
use crate::ray::Ray;
use crate::texture::Texture;

#[derive(Debug, Clone)]
pub struct Material {
    pub albedo: Vector3,        // Color base
    pub specular: f32,          // Reflectividad especular
    pub transparency: f32,      // Transparencia (0.0 = opaco, 1.0 = transparente)
    pub reflectivity: f32,      // Reflectividad (0.0 = no refleja, 1.0 = espejo)
    pub refractive_index: f32,  // Índice de refracción
    pub texture: Option<Texture>,
    pub material_type: MaterialType,
}

#[derive(Debug, Clone, Copy)]
pub enum MaterialType {
    Wood,
    Metal,
    Glass,
    Ceramic,
    Tile,
}

impl Material {
    // Material de madera (para mesas, sillas, mostradores)
    pub fn wood(texture: Option<Texture>) -> Self {
        Material {
            albedo: Vector3::new(0.8, 0.6, 0.4),
            specular: 0.1,
            transparency: 0.0,
            reflectivity: 0.05,
            refractive_index: 1.0,
            texture,
            material_type: MaterialType::Wood,
        }
    }

    // Material metálico (para cafetera - con reflexión)
    pub fn metal(texture: Option<Texture>) -> Self {
        Material {
            albedo: Vector3::new(0.8, 0.8, 0.9),
            specular: 0.9,
            transparency: 0.0,
            reflectivity: 0.8, // Alta reflectividad para efectos de espejo
            refractive_index: 1.0,
            texture,
            material_type: MaterialType::Metal,
        }
    }

    // Material de vidrio (para display - con refracción)
    pub fn glass(texture: Option<Texture>) -> Self {
        Material {
            albedo: Vector3::new(0.9, 0.9, 0.95),
            specular: 0.9,
            transparency: 0.85, // Muy transparente
            reflectivity: 0.1,
            refractive_index: 1.5, // Índice de refracción del vidrio
            texture,
            material_type: MaterialType::Glass,
        }
    }

    // Material cerámico (para tazas, postres)
    pub fn ceramic(texture: Option<Texture>) -> Self {
        Material {
            albedo: Vector3::new(0.95, 0.95, 0.9),
            specular: 0.3,
            transparency: 0.0,
            reflectivity: 0.1,
            refractive_index: 1.0,
            texture,
            material_type: MaterialType::Ceramic,
        }
    }

    // Material de baldosas (para piso)
    pub fn tile(texture: Option<Texture>) -> Self {
        Material {
            albedo: Vector3::new(0.7, 0.7, 0.8),
            specular: 0.4,
            transparency: 0.0,
            reflectivity: 0.2,
            refractive_index: 1.0,
            texture,
            material_type: MaterialType::Tile,
        }
    }

    pub fn get_color(&self, u: f32, v: f32) -> Vector3 {
        if let Some(tex) = &self.texture {
            tex.sample(u, v)
        } else {
            self.albedo
        }
    }

    // Calcular reflexión usando la ley de Snell
    pub fn reflect(&self, incident: Vector3, normal: Vector3) -> Vector3 {
        incident - normal * 2.0 * incident.dot(normal)
    }

    // Calcular refracción usando la ley de Snell
    pub fn refract(&self, incident: Vector3, normal: Vector3, eta_ratio: f32) -> Option<Vector3> {
        let cos_theta = (-incident).dot(normal).min(1.0);
        let r_out_perp = (incident + normal * cos_theta) * eta_ratio;
        let r_out_parallel_squared = 1.0 - r_out_perp.length_squared();
        
        if r_out_parallel_squared < 0.0 {
            None // Reflexión total interna
        } else {
            let r_out_parallel = -normal * r_out_parallel_squared.sqrt();
            Some(r_out_perp + r_out_parallel)
        }
    }

    // Ecuaciones de Fresnel para determinar cuánto se refleja vs refracta
    pub fn fresnel(&self, cos_theta: f32, eta_ratio: f32) -> f32 {
        let r0 = ((1.0 - eta_ratio) / (1.0 + eta_ratio)).powi(2);
        r0 + (1.0 - r0) * (1.0 - cos_theta).powi(5)
    }
}