mod vec3;
mod ray;
mod camera;
mod cube;
mod light;
mod raytracer;
mod material;     
mod skybox;       
mod cafe_scene;   
pub mod texture;

use raylib::prelude::*;
use raytracer::RayTracer;
use std::time::Instant;

const WINDOW_WIDTH: i32 = 800;
const WINDOW_HEIGHT: i32 = 600;

fn main() {
    let (mut rl, thread) = raylib::init()
        .size(WINDOW_WIDTH, WINDOW_HEIGHT)
        .title("Cafetería Ray Tracer - Proyecto de Gráficas")
        .build();

    rl.set_target_fps(30);

    // Create raytracer
    let mut raytracer = RayTracer::new(WINDOW_WIDTH as u32, WINDOW_HEIGHT as u32);
    
    // Variables para animación
    let start_time = Instant::now();
    let mut frame_count = 0u32;
    
    println!("🏪 Iniciando Cafetería Ray Tracer...");
    println!("⚙️  Efectos implementados:");
    println!("   🪨 5 Materiales diferentes (Madera, Metal, Vidrio, Cerámica, Baldosas)");
    println!("   🪞 Reflexión en cafetera metálica");
    println!("   🔍 Refracción en display de vidrio");
    println!("   🌅 Skybox de ambiente cafetería");
    println!("   🎥 Cámara rotativa con zoom");
    
    // Renderizar frame inicial
    let mut current_pixels = raytracer.render_fast();
    let mut texture = rl.load_texture_from_image(&thread, &create_image_from_pixels(&current_pixels, WINDOW_WIDTH, WINDOW_HEIGHT))
        .expect("Failed to create initial texture");

    // Main game loop con animación
    while !rl.window_should_close() {
        let elapsed_time = start_time.elapsed().as_secs_f32();
        
        // Actualizar cámara con animación
        raytracer.update_camera(elapsed_time);
        
        // Renderizar nuevo frame (cada ciertos frames para performance)
        if frame_count % 2 == 0 { // Renderizar cada 2 frames
            let render_start = Instant::now();
            current_pixels = raytracer.render_fast();
            let render_time = render_start.elapsed();
            
            // Actualizar textura - simplificado para evitar problemas de API
            let new_image = create_image_from_pixels(&current_pixels, WINDOW_WIDTH, WINDOW_HEIGHT);
            // En lugar de unload/reload, creamos nueva textura cada vez
            // Esto es menos eficiente pero evita problemas de API
            texture = rl.load_texture_from_image(&thread, &new_image)
                .expect("Failed to update texture");
            
            if frame_count % 60 == 0 {
                println!("Frame {}: Render time: {:.1}ms", frame_count, render_time.as_millis());
            }
        }
        
        let mut d = rl.begin_drawing(&thread);
        
        d.clear_background(Color::BLACK);
        
        // Dibujar la imagen ray-traced
        d.draw_texture(&texture, 0, 0, Color::WHITE);
        
        // UI Information
        d.draw_text(
            "CAFETERÍA RAY TRACER",
            10, 10, 20, Color::WHITE,
        );
        
        d.draw_text(
            &format!("Tiempo: {:.1}s | Frame: {}", elapsed_time, frame_count),
            10, 35, 16, Color::WHITE,
        );
        
        // Efectos implementados
        d.draw_text("EFECTOS ACTIVOS:", 10, WINDOW_HEIGHT - 140, 14, Color::YELLOW);
        d.draw_text("Reflexión (Cafetera)", 10, WINDOW_HEIGHT - 120, 12, Color::WHITE);
        d.draw_text("Refracción (Display)", 10, WINDOW_HEIGHT - 105, 12, Color::WHITE);
        d.draw_text("Skybox Dinámico", 10, WINDOW_HEIGHT - 90, 12, Color::WHITE);
        d.draw_text("Cámara Rotativa", 10, WINDOW_HEIGHT - 75, 12, Color::WHITE);
        d.draw_text("5 Materiales", 10, WINDOW_HEIGHT - 60, 12, Color::WHITE);
        
        // Controles
        d.draw_text("CONTROLES:", 10, WINDOW_HEIGHT - 40, 12, Color::LIME);
        d.draw_text("ESC - Salir", 10, WINDOW_HEIGHT - 25, 11, Color::WHITE);
        
        frame_count += 1;
    }

    // Al finalizar, no necesitamos unload porque raylib lo maneja automáticamente
    println!("🎉 ¡Ray tracer de cafetería finalizado!");
    println!("📈 Total de frames renderizados: {}", frame_count);
    println!("⏱️  Tiempo total de ejecución: {:.1}s", start_time.elapsed().as_secs_f32());
}

fn create_image_from_pixels(pixels: &[u8], width: i32, height: i32) -> Image {
    let mut image = Image::gen_image_color(width, height, Color::BLACK);
    
    unsafe {
        let image_data = std::slice::from_raw_parts_mut(
            image.data as *mut u8,
            (width * height * 4) as usize,
        );
        image_data.copy_from_slice(pixels);
    }
    
    image
}