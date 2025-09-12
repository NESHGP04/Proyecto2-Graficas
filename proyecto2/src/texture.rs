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
}
