// use crate::vec3::Vector3;
// use crate::ray::Ray;

// #[derive(Debug, Clone)]
// pub struct Skybox {
//     pub colors: SkyboxColors,
// }

// #[derive(Debug, Clone)]
// pub struct SkyboxColors {
//     pub ceiling: Vector3,
//     pub walls: Vector3,
//     pub warm_light: Vector3,
//     pub cool_shadow: Vector3,
// }

// impl Skybox {
//     pub fn coffee_shop() -> Self {
//         Skybox {
//             colors: SkyboxColors {
//                 ceiling: Vector3::new(0.7, 0.85, 0.95),     // Celeste pálido (cielo)
//                 walls: Vector3::new(0.8, 0.9, 0.95),       // Azul muy claro
//                 warm_light: Vector3::new(0.9, 0.95, 1.0),  // Blanco azulado
//                 cool_shadow: Vector3::new(0.6, 0.75, 0.85), // Azul grisáceo
//             }
//         }
//     }

//     pub fn sample(&self, ray: &Ray) -> Vector3 {
//         let direction = ray.direction.normalize();
        
//         // Determinar si estamos mirando hacia arriba, abajo o a los lados
//         let y = direction.y;
//         let horizontal_length = (direction.x * direction.x + direction.z * direction.z).sqrt();
        
//         // Crear gradiente basado en la dirección
//         if y > 0.7 {
//             // Mirando hacia el techo
//             let factor = (y - 0.7) / 0.3;
//             self.colors.ceiling * (0.8 + factor * 0.2)
//         } else if y < -0.1 {
//             // Mirando hacia abajo (piso)
//             let factor = (-y - 0.1) / 0.9;
//             self.colors.walls * (0.4 + factor * 0.3)
//         } else {
//             // Mirando horizontalmente (paredes)
//             // Crear variación basada en la dirección horizontal
//             let angle = direction.z.atan2(direction.x);
//             let wave = (angle * 4.0).sin() * 0.1 + (angle * 2.0).cos() * 0.05;
            
//             // Mezclar entre luz cálida y sombra fría
//             let warm_factor = (0.5 + wave).max(0.0).min(1.0);
            
//             let base_color = self.colors.walls * (0.7 + y * 0.2);
//             let warm_component = self.colors.warm_light * warm_factor * 0.3;
//             let cool_component = self.colors.cool_shadow * (1.0 - warm_factor) * 0.2;
            
//             base_color + warm_component + cool_component
//         }
//     }

//     // Skybox alternativo más simple
//     pub fn simple_gradient() -> Self {
//         Skybox {
//             colors: SkyboxColors {
//                 ceiling: Vector3::new(0.8, 0.9, 1.0),
//                 walls: Vector3::new(0.7, 0.8, 0.9),
//                 warm_light: Vector3::new(1.0, 0.95, 0.8),
//                 cool_shadow: Vector3::new(0.6, 0.7, 0.8),
//             }
//         }
//     }

// }

//CON NUBES
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
                ceiling: Vector3::new(0.7, 0.85, 0.95),     // Celeste pálido (cielo)
                walls: Vector3::new(0.8, 0.9, 0.95),       // Azul muy claro
                warm_light: Vector3::new(0.9, 0.95, 1.0),  // Blanco azulado
                cool_shadow: Vector3::new(0.6, 0.75, 0.85), // Azul grisáceo
            }
        }
    }

    // Función para generar ruido simple para las nubes
    fn noise2d(&self, x: f32, y: f32) -> f32 {
        // Ruido simple basado en funciones trigonométricas
        let n1 = (x * 0.1 + y * 0.15).sin();
        let n2 = (x * 0.2 - y * 0.1).sin();
        let n3 = (x * 0.05 + y * 0.05).sin();
        let n4 = (x * 0.3 + y * 0.25).cos();
        
        // Combinar diferentes frecuencias para textura más rica
        (n1 + n2 * 0.5 + n3 * 0.25 + n4 * 0.15) / 1.9
    }

    // Función adicional para variación de nubes
    fn cloud_pattern(&self, x: f32, y: f32) -> f32 {
        let base = self.noise2d(x, y);
        let detail = self.noise2d(x * 2.5, y * 2.5) * 0.3;
        (base + detail).max(-0.2).min(0.8)
    }

    pub fn sample(&self, ray: &Ray) -> Vector3 {
        let direction = ray.direction.normalize();
        
        // Determinar si estamos mirando hacia arriba, abajo o a los lados
        let y = direction.y;
        
        // Crear gradiente basado en la dirección
        if y > 0.3 {
            // Mirando hacia el cielo - AQUÍ VAN LAS NUBES
            let sky_factor = ((y - 0.3) / 0.7).max(0.0).min(1.0);
            let base_sky = self.colors.ceiling * (0.8 + sky_factor * 0.2);
            
            // === GENERAR NUBES ===
            let cloud_noise = self.cloud_pattern(direction.x * 12.0, direction.z * 12.0);
            let cloud_factor = ((cloud_noise + 0.4) * 1.8).max(0.0).min(0.9);
            
            // Colores de las nubes
            let cloud_white = Vector3::new(1.0, 1.0, 1.0);           // Blanco puro
            let cloud_shadow = Vector3::new(0.85, 0.88, 0.92);      // Gris azulado claro
            let cloud_dark = Vector3::new(0.7, 0.75, 0.82);         // Gris más oscuro para profundidad
            
            // Crear variación en las nubes (sombras y luces)
            let cloud_variation = self.noise2d(direction.x * 20.0, direction.z * 20.0);
            let shadow_factor = ((cloud_variation + 0.2) * 1.2).max(0.0).min(1.0);
            
            // Mezclar colores de nube
            let cloud_color = if shadow_factor > 0.6 {
                cloud_white                                          // Partes iluminadas
            } else if shadow_factor > 0.3 {
                cloud_shadow                                         // Partes intermedias
            } else {
                cloud_dark                                           // Sombras de las nubes
            };
            
            // Combinar cielo base con nubes
            let cloud_opacity = cloud_factor * 0.7; // Las nubes no son completamente opacas
            base_sky * (1.0 - cloud_opacity) + cloud_color * cloud_opacity
            
        } else if y < -0.1 {
            // Mirando hacia abajo (piso/suelo)
            let factor = ((-y - 0.1) / 0.9).max(0.0).min(1.0);
            self.colors.walls * (0.4 + factor * 0.3)
        } else {
            // Mirando horizontalmente (horizonte)
            // Crear gradiente suave hacia el horizonte
            let horizon_factor = (y + 0.1) / 0.4; // De -0.1 a 0.3
            let horizon_color = self.colors.ceiling * 0.9 + self.colors.walls * 0.1;
            
            // Agregar algo de variación atmosférica
            let angle = direction.z.atan2(direction.x);
            let atmosphere_variation = (angle * 3.0).sin() * 0.05 + (angle * 1.5).cos() * 0.03;
            let varied_color = horizon_color + Vector3::new(atmosphere_variation, atmosphere_variation * 0.5, -atmosphere_variation * 0.3);
            
            // Transición suave entre cielo y "paredes"
            let base_color = self.colors.walls * (0.7 + y * 0.2);
            let warm_component = self.colors.warm_light * horizon_factor * 0.2;
            let cool_component = self.colors.cool_shadow * (1.0 - horizon_factor) * 0.15;
            
            base_color + warm_component + cool_component + varied_color * 0.3
        }
    }

    // Skybox alternativo sin nubes (por si acaso)
    pub fn simple_sky() -> Self {
        Skybox {
            colors: SkyboxColors {
                ceiling: Vector3::new(0.7, 0.85, 0.95),     // Celeste pálido
                walls: Vector3::new(0.8, 0.9, 0.95),       // Azul muy claro
                warm_light: Vector3::new(0.9, 0.95, 1.0),  // Blanco azulado
                cool_shadow: Vector3::new(0.6, 0.75, 0.85), // Azul grisáceo
            }
        }
    }

    // Skybox con más contraste en las nubes
    pub fn dramatic_sky() -> Self {
        Skybox {
            colors: SkyboxColors {
                ceiling: Vector3::new(0.6, 0.8, 0.95),      // Azul más intenso
                walls: Vector3::new(0.75, 0.85, 0.9),      // Azul grisáceo
                warm_light: Vector3::new(1.0, 0.95, 0.85), // Luz dorada
                cool_shadow: Vector3::new(0.5, 0.65, 0.8),  // Azul más oscuro
            }
        }
    }
}