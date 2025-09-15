use crate::vec3::Vector3;
use crate::cube::{Cube, Plane};
use crate::material::Material;
use crate::texture::Texture;

pub struct CafeScene {
    pub cubes: Vec<Cube>,
    pub ground: Plane,
}

impl CafeScene {
    pub fn new() -> Self {
        let mut cubes = Vec::new();

        // === MATERIALES ===
        
        // Material de madera para muebles
        let wood_material = Material::wood(Some(Texture::load("../assets/wood.png")));
        
        // Material metálico para cafetera (con alta reflectividad)
        let metal_material = Material::metal(Some(Texture::load("../assets/metal.png")));
        
        // Material de vidrio para display (con refracción)
        let glass_material = Material::glass(None);
        
        // Material cerámico para tazas y postres
        let ceramic_material = Material::ceramic(Some(Texture::load("../assets/ceramic.png")));
        
        // Material de baldosas para piso
        let tile_material = Material::tile(Some(Texture::load("../assets/tiles.png")));

        // === CONSTRUCCIÓN DE LA CAFETERÍA ===

        // === MESA 1 (Izquierda) ===
        // Tablero de mesa 1
        cubes.push(Cube::new(
            Vector3::new(-3.0, 0.7, -1.0),
            Vector3::new(-1.0, 0.8, 1.0),
            wood_material.clone(),
        ));
        
        // Patas de mesa 1 (4 patas)
        let table1_leg_positions = [
            (-2.8, -1.8), (-1.2, -1.8), (-2.8, 0.8), (-1.2, 0.8)
        ];
        for &(x, z) in &table1_leg_positions {
            cubes.push(Cube::new(
                Vector3::new(x, 0.0, z),
                Vector3::new(x + 0.1, 0.7, z + 0.1),
                wood_material.clone(),
            ));
        }

        // Sillas mesa 1
        // Silla 1A (frente)
        cubes.push(Cube::new(
            Vector3::new(-2.5, 0.0, -2.2),
            Vector3::new(-1.5, 0.4, -1.8),
            wood_material.clone(),
        ));
        // Respaldo silla 1A
        cubes.push(Cube::new(
            Vector3::new(-2.5, 0.4, -2.2),
            Vector3::new(-1.5, 1.0, -2.1),
            wood_material.clone(),
        ));

        // Silla 1B (atrás)
        cubes.push(Cube::new(
            Vector3::new(-2.5, 0.0, 1.8),
            Vector3::new(-1.5, 0.4, 2.2),
            wood_material.clone(),
        ));
        // Respaldo silla 1B
        cubes.push(Cube::new(
            Vector3::new(-2.5, 0.4, 2.1),
            Vector3::new(-1.5, 1.0, 2.2),
            wood_material.clone(),
        ));

        // Objetos en mesa 1 (cafés y postres)
        // Taza de café 1
        cubes.push(Cube::new(
            Vector3::new(-2.5, 0.8, -0.3),
            Vector3::new(-2.2, 1.0, 0.0),
            ceramic_material.clone(),
        ));
        
        // Postre 1
        cubes.push(Cube::new(
            Vector3::new(-1.8, 0.8, 0.3),
            Vector3::new(-1.5, 0.95, 0.6),
            ceramic_material.clone(),
        ));

        // === MESA 2 (Derecha) ===
        // Tablero de mesa 2
        cubes.push(Cube::new(
            Vector3::new(1.0, 0.7, -1.0),
            Vector3::new(3.0, 0.8, 1.0),
            wood_material.clone(),
        ));
        
        // Patas de mesa 2
        let table2_leg_positions = [
            (1.2, -1.8), (2.8, -1.8), (1.2, 0.8), (2.8, 0.8)
        ];
        for &(x, z) in &table2_leg_positions {
            cubes.push(Cube::new(
                Vector3::new(x, 0.0, z),
                Vector3::new(x + 0.1, 0.7, z + 0.1),
                wood_material.clone(),
            ));
        }

        // Sillas mesa 2
        // Silla 2A
        cubes.push(Cube::new(
            Vector3::new(1.5, 0.0, -2.2),
            Vector3::new(2.5, 0.4, -1.8),
            wood_material.clone(),
        ));
        cubes.push(Cube::new(
            Vector3::new(1.5, 0.4, -2.2),
            Vector3::new(2.5, 1.0, -2.1),
            wood_material.clone(),
        ));

        // Silla 2B
        cubes.push(Cube::new(
            Vector3::new(1.5, 0.0, 1.8),
            Vector3::new(2.5, 0.4, 2.2),
            wood_material.clone(),
        ));
        cubes.push(Cube::new(
            Vector3::new(1.5, 0.4, 2.1),
            Vector3::new(2.5, 1.0, 2.2),
            wood_material.clone(),
        ));

        // Objetos en mesa 2
        // Taza de café 2
        cubes.push(Cube::new(
            Vector3::new(1.5, 0.8, -0.3),
            Vector3::new(1.8, 1.0, 0.0),
            ceramic_material.clone(),
        ));
        
        // Postre 2
        cubes.push(Cube::new(
            Vector3::new(2.2, 0.8, 0.3),
            Vector3::new(2.5, 0.95, 0.6),
            ceramic_material.clone(),
        ));

        // === MOSTRADOR PRINCIPAL (Centro-derecha) ===
        // Base del mostrador
        cubes.push(Cube::new(
            Vector3::new(-0.5, 0.0, 2.5),
            Vector3::new(2.0, 1.0, 3.5),
            wood_material.clone(),
        ));

        // Display de vidrio encima del mostrador (CON REFRACCIÓN)
        cubes.push(Cube::new(
            Vector3::new(0.0, 1.0, 2.7),
            Vector3::new(1.5, 1.8, 3.3),
            glass_material.clone(),
        ));

        // Postres dentro del display
        cubes.push(Cube::new(
            Vector3::new(0.2, 1.1, 2.8),
            Vector3::new(0.5, 1.25, 3.1),
            ceramic_material.clone(),
        ));
        cubes.push(Cube::new(
            Vector3::new(0.8, 1.1, 2.9),
            Vector3::new(1.1, 1.3, 3.2),
            ceramic_material.clone(),
        ));
        cubes.push(Cube::new(
            Vector3::new(1.2, 1.1, 2.8),
            Vector3::new(1.4, 1.2, 3.0),
            ceramic_material.clone(),
        ));

        // === MOSTRADOR DE FONDO ===
        // Base del mostrador trasero
        cubes.push(Cube::new(
            Vector3::new(-2.0, 0.0, 4.0),
            Vector3::new(2.0, 1.2, 4.5),
            wood_material.clone(),
        ));

        // Cafetera encima del mostrador (CON REFLEXIÓN)
        // Base de la cafetera
        cubes.push(Cube::new(
            Vector3::new(-0.8, 1.2, 4.1),
            Vector3::new(0.8, 1.4, 4.4),
            metal_material.clone(),
        ));
        
        // Cuerpo principal de la cafetera
        cubes.push(Cube::new(
            Vector3::new(-0.6, 1.4, 4.15),
            Vector3::new(0.6, 2.2, 4.35),
            metal_material.clone(),
        ));

        // Tapa de la cafetera
        cubes.push(Cube::new(
            Vector3::new(-0.5, 2.2, 4.2),
            Vector3::new(0.5, 2.4, 4.3),
            metal_material.clone(),
        ));

        // === PAREDES ===
        // Pared trasera
        cubes.push(Cube::new(
            Vector3::new(-4.0, 0.0, 4.8),
            Vector3::new(4.0, 3.0, 5.0),
            Material::wood(Some(Texture::load("../assets/wall.png"))),
        ));

        // Paredes laterales
        cubes.push(Cube::new(
            Vector3::new(-4.0, 0.0, -3.0),
            Vector3::new(-3.8, 3.0, 5.0),
            Material::wood(Some(Texture::load("../assets/wall.png"))),
        ));
        
        cubes.push(Cube::new(
            Vector3::new(3.8, 0.0, -3.0),
            Vector3::new(4.0, 3.0, 5.0),
            Material::wood(Some(Texture::load("../assets/wall.png"))),
        ));

        // === PISO ===
        let ground = Plane::new(
            Vector3::new(0.0, 0.0, 0.0),
            Vector3::new(0.0, 1.0, 0.0),
            tile_material,
        );

        CafeScene { cubes, ground }
    }

    // Obtener todos los objetos de la escena para ray tracing
    pub fn get_objects(&self) -> &Vec<Cube> {
        &self.cubes
    }

    pub fn get_ground(&self) -> &Plane {
        &self.ground
    }
}