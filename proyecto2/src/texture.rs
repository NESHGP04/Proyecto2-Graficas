use image::{GenericImageView, RgbaImage};

#[derive(Debug, Clone)]
pub struct Texture {
    pub image: RgbaImage,
    pub width: u32,
    pub height: u32,
}

impl Texture {
    pub fn load(filename: &str) -> Self {
        let img = image::open(filename).expect("Failed to load texture").to_rgba8();
        let (width, height) = img.dimensions();
        Texture { image: img, width, height }
    }

    pub fn sample(&self, u: f32, v: f32) -> crate::vec3::Vector3 {
        let x = ((u.fract() * self.width as f32) as u32).min(self.width - 1);
        let y = (((1.0 - v.fract()) * self.height as f32) as u32).min(self.height - 1);

        let pixel = self.image.get_pixel(x, y);
        crate::vec3::Vector3::new(
            pixel[0] as f32 / 255.0,
            pixel[1] as f32 / 255.0,
            pixel[2] as f32 / 255.0,
        )
    }

    pub fn create_wood(size: u32) -> Self {
        let mut img = RgbaImage::new(size, size);
        for y in 0..size {
            for x in 0..size {
                let wave = ((y as f32 / 10.0).sin() * 20.0) as u32;
                let brown = if (x + wave) % 40 < 20 { 
                    [139, 69, 19, 255] // Marrón oscuro
                } else { 
                    [210, 180, 140, 255] // Marrón claro
                };
                img.put_pixel(x, y, image::Rgba(brown));
            }
        }
        Texture { image: img, width: size, height: size }
    }
    
    pub fn create_metal(size: u32) -> Self {
        let mut img = RgbaImage::new(size, size);
        for y in 0..size {
            for x in 0..size {
                let noise = ((x + y) % 8) as f32 / 8.0;
                let gray = (200.0 + noise * 40.0) as u8;
                img.put_pixel(x, y, image::Rgba([gray, gray, gray + 10, 255]));
            }
        }
        Texture { image: img, width: size, height: size }
    }

    pub fn create_ceramic(size: u32) -> Self {
        let mut img = RgbaImage::new(size, size);
        for y in 0..size {
            for x in 0..size {
                // Base crema/blanca
                let mut base = 245u8; // #F5F5DC aproximado
                // Pequeños detalles aleatorios
                if (x + y) % 50 == 0 { base = 255; } // puntitos blancos
                img.put_pixel(x, y, image::Rgba([base, base, base, 255]));
            }
        }
        Texture { image: img, width: size, height: size }
    }

    pub fn create_tiles(size: u32) -> Self {
        let mut img = RgbaImage::new(size, size);
        let tile_count = 8; // 8x8 baldosas
        let tile_size = size / tile_count;
        for y in 0..size {
            for x in 0..size {
                let in_line = x % tile_size == 0 || y % tile_size == 0;
                if in_line {
                    // línea de lechada (blanco/beige)
                    img.put_pixel(x, y, image::Rgba([245, 245, 220, 255])); // #F5F5DC
                } else {
                    // baldosas (gris claro)
                    img.put_pixel(x, y, image::Rgba([211, 211, 211, 255])); // #D3D3D3
                }
            }
        }
        Texture { image: img, width: size, height: size }
    }

    pub fn create_walls(size: u32) -> Self {
        let mut img = RgbaImage::new(size, size);
        for y in 0..size {
            for x in 0..size {
                // Tonos cálidos base
                let r = 222u8; // #DEB887
                let g = 184u8;
                let b = 135u8;
                // Variación sutil aleatoria
                let variation = ((x + y) % 20) as u8; 
                img.put_pixel(x, y, image::Rgba([r.saturating_add(variation), g.saturating_add(variation/2), b, 255]));
            }
        }
        Texture { image: img, width: size, height: size }
    }
}
