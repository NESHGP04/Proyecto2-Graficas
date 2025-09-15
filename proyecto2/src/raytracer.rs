use crate::vec3::Vector3;
use crate::ray::{Ray, HitRecord};
use crate::camera::Camera;
use crate::cube::{Cube, Plane};
use crate::light::{DirectionalLight, AmbientLight};
use crate::material::Material;
use crate::skybox::Skybox;
use crate::cafe_scene::CafeScene;

pub struct Scene {
    pub cafe_scene: CafeScene,
    pub directional_light: DirectionalLight,
    pub ambient_light: AmbientLight,
    pub skybox: Skybox,
}

impl Scene {
    pub fn new() -> Self {
        Scene {
            cafe_scene: CafeScene::new(),
            // Luz direccional desde arriba-izquierda
            directional_light: DirectionalLight::new(
                Vector3::new(1.0, -1.0, -1.0),
                Vector3::new(1.0, 0.95, 0.8), // Luz cálida de cafetería
                0.7,
            ),
            // Luz ambiental suave
            ambient_light: AmbientLight::new(
                Vector3::new(1.0, 0.9, 0.8),
                0.3,
            ),
            // Skybox de cafetería
            skybox: Skybox::coffee_shop(),
        }
    }

    fn hit(&self, ray: &Ray, t_min: f32, t_max: f32) -> Option<HitRecord> {
        let mut closest_hit: Option<HitRecord> = None;
        let mut closest_t = t_max;

        // Check all cubes in the cafe scene
        for cube in self.cafe_scene.get_objects() {
            if let Some(hit) = cube.hit(ray, t_min, closest_t) {
                closest_t = hit.t;
                closest_hit = Some(hit);
            }
        }

        // Check ground
        if let Some(hit) = self.cafe_scene.get_ground().hit(ray, t_min, closest_t) {
            closest_t = hit.t;
            closest_hit = Some(hit);
        }

        closest_hit
    }

    fn is_in_shadow(&self, point: Vector3, light_direction: Vector3) -> bool {
        let shadow_ray = Ray::new(point + light_direction * 0.001, light_direction);
        
        // Check shadows against all objects
        for cube in self.cafe_scene.get_objects() {
            if let Some(_) = cube.hit(&shadow_ray, 0.001, f32::INFINITY) {
                return true;
            }
        }

        if let Some(_) = self.cafe_scene.get_ground().hit(&shadow_ray, 0.001, f32::INFINITY) {
            return true;
        }

        false
    }

    fn calculate_lighting(&self, hit: &HitRecord, view_direction: Vector3, ray: &Ray, depth: u32) -> Vector3 {
        let mut final_color = Vector3::zero();
        
        // Base lighting (ambient + diffuse)
        let ambient = self.ambient_light.calculate_ambient(hit.material_color);
        
        let in_shadow = self.is_in_shadow(hit.point, self.directional_light.direction);
        let diffuse = if in_shadow {
            self.directional_light.calculate_diffuse(hit.normal, hit.material_color) * 0.2
        } else {
            self.directional_light.calculate_diffuse(hit.normal, hit.material_color)
        };
        
        final_color = ambient + diffuse;

        // Advanced material effects
        if let Some(material) = &hit.material {
            // === REFLEXIÓN ===
            if material.reflectivity > 0.0 && depth < 3 {
                let reflect_direction = material.reflect(-view_direction, hit.normal);
                let reflect_ray = Ray::new(hit.point + hit.normal * 0.001, reflect_direction);
                let reflected_color = self.trace_ray_internal(&reflect_ray, depth + 1);
                final_color = final_color * (1.0 - material.reflectivity) + 
                            reflected_color * material.reflectivity;
            }

            // === REFRACCIÓN (solo para materiales transparentes) ===
            if material.transparency > 0.0 && depth < 3 {
                let eta_ratio = if hit.front_face {
                    1.0 / material.refractive_index
                } else {
                    material.refractive_index
                };

                let cos_theta = (-view_direction).dot(hit.normal).min(1.0);
                let sin_theta = (1.0 - cos_theta * cos_theta).sqrt();

                // Comprobar si hay reflexión total interna
                let cannot_refract = eta_ratio * sin_theta > 1.0;
                let fresnel_factor = material.fresnel(cos_theta, eta_ratio);

                if !cannot_refract && fresnel_factor < 0.9 {
                    if let Some(refract_direction) = material.refract(view_direction, hit.normal, eta_ratio) {
                        let refract_ray = Ray::new(hit.point - hit.normal * 0.001, refract_direction);
                        let refracted_color = self.trace_ray_internal(&refract_ray, depth + 1);
                        
                        // Mezclar color refractado con el material
                        final_color = final_color * (1.0 - material.transparency) + 
                                    refracted_color * material.transparency * (1.0 - fresnel_factor);
                    }
                }
            }

            // === EFECTOS ESPECULARES ===
            if material.specular > 0.0 {
                let light_reflect = material.reflect(self.directional_light.direction, hit.normal);
                let spec_dot = light_reflect.dot(-view_direction).max(0.0);
                let spec_power = spec_dot.powf(32.0); // Shininess
                let specular_color = Vector3::one() * material.specular * spec_power * 0.5;
                final_color = final_color + specular_color;
            }
        }

        final_color
    }

    fn trace_ray_internal(&self, ray: &Ray, depth: u32) -> Vector3 {
        if depth >= 5 {
            return self.skybox.sample(ray);
        }

        if let Some(hit) = self.hit(ray, 0.001, f32::INFINITY) {
            self.calculate_lighting(&hit, ray.direction, ray, depth)
        } else {
            self.skybox.sample(ray)
        }
    }
}

pub struct RayTracer {
    pub width: u32,
    pub height: u32,
    pub camera: Camera,
    pub scene: Scene,
    pub animation_time: f32,
}

impl RayTracer {
    pub fn new(width: u32, height: u32) -> Self {
        let aspect_ratio = width as f32 / height as f32;
        
        // Posición inicial de la cámara para ver toda la cafetería
        let camera = Camera::new(
            Vector3::new(4.0, 2.5, -2.0),  // Posición elevada
            Vector3::new(0.0, 1.0, 1.0),   // Mirando hacia el centro
            Vector3::new(0.0, 1.0, 0.0),   // Vector "up"
            60.0,
            aspect_ratio,
        );

        RayTracer {
            width,
            height,
            camera,
            scene: Scene::new(),
            animation_time: 0.0,
        }
    }

    pub fn update_camera(&mut self, time: f32) {
        self.animation_time = time;
        
        // Rotación de cámara alrededor de la cafetería
        let radius = 5.0;
        let height = 2.5;
        let speed = 0.3;
        
        let angle = time * speed;
        let x = angle.cos() * radius;
        let z = angle.sin() * radius;
        
        // Oscilación suave en altura
        let height_oscillation = (time * 0.5).sin() * 0.3;
        let camera_height = height + height_oscillation;
        
        // Zoom in/out suave
        let zoom_factor = 1.0 + (time * 0.4).sin() * 0.2;
        let camera_distance = radius * zoom_factor;
        
        let final_x = angle.cos() * camera_distance;
        let final_z = angle.sin() * camera_distance;
        
        // Actualizar posición y target de la cámara
        let aspect_ratio = self.width as f32 / self.height as f32;
        self.camera = Camera::new(
            Vector3::new(final_x, camera_height, final_z),
            Vector3::new(0.0, 1.0, 1.0), // Siempre mirando hacia el centro de la cafetería
            Vector3::new(0.0, 1.0, 0.0),
            60.0 + (time * 0.6).sin() * 5.0, // FOV que varía ligeramente
            aspect_ratio,
        );
    }

    pub fn trace_ray(&self, ray: &Ray) -> Vector3 {
        self.scene.trace_ray_internal(ray, 0)
    }

    pub fn render(&self) -> Vec<u8> {
        let mut pixels = vec![0u8; (self.width * self.height * 4) as usize];
        
        for y in 0..self.height {
            for x in 0..self.width {
                let u = x as f32 / (self.width - 1) as f32;
                let v = (self.height - 1 - y) as f32 / (self.height - 1) as f32;
                
                let ray = self.camera.get_ray(u, v);
                let color = self.trace_ray(&ray);
                
                // Tone mapping simple para evitar colores sobre-saturados
                let mapped_color = Vector3::new(
                    color.x / (1.0 + color.x),
                    color.y / (1.0 + color.y),
                    color.z / (1.0 + color.z),
                );
                
                // Clamp and convert to 0-255 range
                let r = (mapped_color.x.min(1.0).max(0.0) * 255.0) as u8;
                let g = (mapped_color.y.min(1.0).max(0.0) * 255.0) as u8;
                let b = (mapped_color.z.min(1.0).max(0.0) * 255.0) as u8;
                
                let pixel_index = ((y * self.width + x) * 4) as usize;
                pixels[pixel_index] = r;
                pixels[pixel_index + 1] = g;
                pixels[pixel_index + 2] = b;
                pixels[pixel_index + 3] = 255;
            }
            
            // Print progress menos frecuentemente para mejor performance
            if y % (self.height / 5) == 0 {
                println!("Rendering: {}%", (y * 100) / self.height);
            }
        }
        
        println!("Rendering: 100%");
        pixels
    }

    // Versión optimizada para animación en tiempo real
    pub fn render_fast(&self) -> Vec<u8> {
        let mut pixels = vec![0u8; (self.width * self.height * 4) as usize];
        let scale = 2; // Renderizar a menor resolución para mejor performance
        
        let render_width = self.width / scale;
        let render_height = self.height / scale;
        
        for y in 0..render_height {
            for x in 0..render_width {
                let u = x as f32 / (render_width - 1) as f32;
                let v = (render_height - 1 - y) as f32 / (render_height - 1) as f32;
                
                let ray = self.camera.get_ray(u, v);
                let color = self.trace_ray(&ray);
                
                let mapped_color = Vector3::new(
                    color.x / (1.0 + color.x),
                    color.y / (1.0 + color.y),
                    color.z / (1.0 + color.z),
                );
                
                let r = (mapped_color.x.min(1.0).max(0.0) * 255.0) as u8;
                let g = (mapped_color.y.min(1.0).max(0.0) * 255.0) as u8;
                let b = (mapped_color.z.min(1.0).max(0.0) * 255.0) as u8;
                
                // Upscale por duplicación de píxeles
                for dy in 0..scale {
                    for dx in 0..scale {
                        let final_x = x * scale + dx;
                        let final_y = y * scale + dy;
                        
                        if final_x < self.width && final_y < self.height {
                            let pixel_index = ((final_y * self.width + final_x) * 4) as usize;
                            pixels[pixel_index] = r;
                            pixels[pixel_index + 1] = g;
                            pixels[pixel_index + 2] = b;
                            pixels[pixel_index + 3] = 255;
                        }
                    }
                }
            }
        }
        
        pixels
    }
}