# Cafetería Ray Tracer

**Proyecto de Gráficas por Computadora**  
*Ray Tracer implementado en Rust con efectos avanzados de iluminación*

## Descripción del Proyecto

Este proyecto implementa un ray tracer completo en Rust que renderiza una cafetería en 3D con efectos realistas de iluminación, reflexión, refracción y materiales avanzados. La escena incluye mobiliario detallado, objetos decorativos y un sistema de cámara animada que rota alrededor de la escena.

## Fotografías de Demostración

*
<img width="795" height="594" alt="Captura de pantalla 2025-09-22 a la(s) 9 07 53 a  m" src="https://github.com/user-attachments/assets/47c3438b-3fc4-4c42-8838-feb8ae1dd6c1" />
<img width="791" height="603" alt="Captura de pantalla 2025-09-16 a la(s) 9 53 32 p  m" src="https://github.com/user-attachments/assets/e8e497ad-ccc8-4dcb-a179-b0b44e4807ba" />
<img width="803" height="603" alt="Captura de pantalla 2025-09-16 a la(s) 8 44 39 a  m" src="https://github.com/user-attachments/assets/9def1645-7b74-413a-81d8-81139f0eb970" />
*

## Características Implementadas

### Escena Completa (30 puntos)
- **2 mesas de madera** con objetos encima (tazas de café y postres)
- **4 sillas** (2 por mesa) con respaldos
- **Mostrador principal** con display de vidrio para postres
- **Mostrador trasero** con cafetera metálica
- **Arquitectura**: Paredes, piso con baldosas
- **Objetos decorativos**: Múltiples postres, tazas, elementos de cafetería

### Sistema de Materiales Avanzados (30 puntos - 6 materiales)
1. **Madera** - Mesas, sillas, mostradores y paredes
2. **Metal** - Cafetera con alta reflectividad
3. **Vidrio** - Display transparente con refracción
4. **Cerámica** - Tazas y algunos postres
5. **Baldosas** - Piso de la cafetería
6. **Chocolate** - Postres especializados

### Efectos de Iluminación y Materiales
- **Reflexión (5 puntos)** - Implementada en la cafetera metálica
- **Refracción (10 puntos)** - Implementada en el display de vidrio
- **Skybox dinámico (20 puntos)** - Cielo con nubes procedurales
- **Iluminación avanzada** - Luz direccional y ambiental con sombras

### Sistema de Cámara Animada (20 puntos)
- **Rotación orbital** - La cámara gira alrededor de la cafetería
- **Zoom dinámico** - Acercamiento y alejamiento automático
- **Variación de altura** - Movimiento vertical suave
- **FOV variable** - Campo de visión que cambia sutilmente

### Características Técnicas Adicionales
- **Ray tracing recursivo** para reflexiones y refracciones múltiples
- **Sistema de sombras** suaves y realistas
- **Mapeo de texturas UV** para todos los objetos
- **Tone mapping** para evitar sobre-saturación de colores
- **Renderizado optimizado** con modo rápido para animación en tiempo real

## Estructura del Proyecto

```
cafe_raytracer/
├── src/
│   ├── main.rs           # Loop principal con animación
│   ├── raytracer.rs      # Motor de ray tracing
│   ├── camera.rs         # Sistema de cámara
│   ├── material.rs       # Sistema de materiales avanzados
│   ├── cube.rs           # Geometría de cubos con materiales
│   ├── cafe_scene.rs     # Construcción de la cafetería
│   ├── skybox.rs         # Sistema de skybox con nubes
│   ├── light.rs          # Sistema de iluminación
│   ├── texture.rs        # Manejo de texturas
│   ├── ray.rs            # Estructuras de ray y hit
│   └── vec3.rs           # Matemáticas vectoriales
├── assets/
│   ├── wood.png          # Textura de madera
│   ├── metal.png         # Textura metálica
│   ├── ceramic.png       # Textura cerámica
│   ├── tiles.png         # Textura de baldosas
│   ├── chocolate.png     # Textura de chocolate
│   └── wall.png          # Textura de paredes
├── Cargo.toml
└── README.md
```

## Dependencias

```toml
[dependencies]
raylib = "3.7"
image = "0.24"
```

## Compilación y Ejecución

```bash
# Clonar el repositorio
git clone [URL_DEL_REPOSITORIO]
cd cafe_raytracer

# Compilar y ejecutar
cargo run --release
```

## Controles

- **ESC** - Salir de la aplicación
- La cámara se anima automáticamente

## Detalles Técnicos

### Ray Tracing
- **Algoritmo**: Ray casting con intersección AABB para cubos
- **Recursividad**: Hasta 5 niveles para efectos múltiples
- **Performance**: Renderizado adaptativo con escalado dinámico

### Física de Materiales
- **Reflexión**: Ley de reflexión con coeficientes variables
- **Refracción**: Ley de Snell con índices de refracción reales
- **Ecuaciones de Fresnel**: Para determinar reflexión vs refracción
- **Sombreado**: Modelo de iluminación Lambertiano con componente especular

### Optimizaciones
- **Renderizado escalonado**: Menor resolución para animación fluida
- **Culling temprano**: Eliminación de rayos que no contribuyen
- **Reutilización de cálculos**: Cache de valores frecuentemente usados

## Autor

**Marinés García**  
*Proyecto de Gráficas por Computadora*  
*28 de septiembre del 2025*

## Notas de Desarrollo

Este proyecto fue desarrollado completamente en Rust sin librerías externas de ray tracing, implementando todos los algoritmos desde cero. La escena representa una cafetería funcional con elementos realistas y efectos de iluminación avanzados que demuestran un entendimiento profundo de las técnicas de rendering 3D.

El código está estructurado de manera modular para facilitar extensiones futuras y modificaciones de la escena o efectos de rendering.
