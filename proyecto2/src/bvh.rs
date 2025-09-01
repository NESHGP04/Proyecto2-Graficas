// - Simple BVH-like spatial grouping (optional naive split)

use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Instant;
use std::ops::{Add, Sub, Mul, Div, Neg};

pub struct BVHNode {
    // campos del nodo BVH
}

// Métodos de BVHNode
impl BVHNode {
    pub fn new() -> Self {
        Self { }
    }
}
