use crate::vec3::Vector3;
use crate::ray::Ray;

#[derive(Debug, Clone)]
pub struct Skybox {
    pub colors: SkyboxColors,
}

#[derive(Debug, Clone)]
pub struct SkyboxColors {
    pub ceiling: Vector3,
    pub walls: Vector3,
    pub warm_light: Vector3,
    pub cool_shadow: Vector3,
}

impl Skybox {
    pub fn coffee_shop() -> Self {
        Skybox {
            colors: SkyboxColors {
                ceiling: Vector3::new(0.9, 0.85, 0.7),      // Beige claro
                walls: Vector3::new(0.8, 0.7, 0.5),        // Marrón cálido
                warm_light: Vector3::new(1.0, 0.9, 0.7),   // Luz cálida
                cool_shadow: Vector3::new(0.6, 0.65, 0.7), // Sombra fría
            }
        }
    }

    pub fn sample(&self, ray: &Ray) -> Vector3 {
        let direction = ray.direction.normalize();
        
        // Determinar si estamos mirando hacia arriba, abajo o a los lados
        let y = direction.y;
        let horizontal_length = (direction.x * direction.x + direction.z * direction.z).sqrt();
        
        // Crear gradiente basado en la dirección
        if y > 0.7 {
            // Mirando hacia el techo
            let factor = (y - 0.7) / 0.3;
            self.colors.ceiling * (0.8 + factor * 0.2)
        } else if y < -0.1 {
            // Mirando hacia abajo (piso)
            let factor = (-y - 0.1) / 0.9;
            self.colors.walls * (0.4 + factor * 0.3)
        } else {
            // Mirando horizontalmente (paredes)
            // Crear variación basada en la dirección horizontal
            let angle = direction.z.atan2(direction.x);
            let wave = (angle * 4.0).sin() * 0.1 + (angle * 2.0).cos() * 0.05;
            
            // Mezclar entre luz cálida y sombra fría
            let warm_factor = (0.5 + wave).max(0.0).min(1.0);
            
            let base_color = self.colors.walls * (0.7 + y * 0.2);
            let warm_component = self.colors.warm_light * warm_factor * 0.3;
            let cool_component = self.colors.cool_shadow * (1.0 - warm_factor) * 0.2;
            
            base_color + warm_component + cool_component
        }
    }

    // Skybox alternativo más simple
    pub fn simple_gradient() -> Self {
        Skybox {
            colors: SkyboxColors {
                ceiling: Vector3::new(0.8, 0.9, 1.0),
                walls: Vector3::new(0.7, 0.8, 0.9),
                warm_light: Vector3::new(1.0, 0.95, 0.8),
                cool_shadow: Vector3::new(0.6, 0.7, 0.8),
            }
        }
    }
}