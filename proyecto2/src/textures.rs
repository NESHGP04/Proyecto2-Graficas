//image loader, procedural
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Instant;
use std::ops::{Add, Sub, Mul, Div, Neg};

use crate::core::Vec3;

// ---------------------------- Texture loader (PPM P3) ----------------------------
pub struct Texture { pub w: usize, pub h: usize, pub data: Vec<u8> } // RGB bytes

impl Texture {
    pub fn load_ppm(path: &str)->Option<Self> {
        let f = File::open(path).ok()?;
        let mut reader = BufReader::new(f);
        let mut header = String::new();
        reader.read_line(&mut header).ok()?;
        if !header.trim().starts_with("P3") { return None; }
        // skip comments
        let mut dims_line = String::new();
        loop {
            dims_line.clear();
            reader.read_line(&mut dims_line).ok()?;
            if dims_line.trim().starts_with('#') { continue } else { break }
        }
        let mut parts = dims_line.split_whitespace();
        let w: usize = parts.next()?.parse().ok()?;
        let h: usize = parts.next()?.parse().ok()?;
        let mut max_line = String::new(); reader.read_line(&mut max_line).ok()?; // max color
        let mut nums = Vec::new();
        for line in reader.lines() { if let Ok(l) = line { for tok in l.split_whitespace() { nums.push(tok.to_string()) } } }
        if nums.len() < w*h*3 { return None; }
        let mut data = Vec::with_capacity(w*h*3);
        for i in 0..(w*h*3) { let v: u8 = nums[i].parse::<u32>().ok().map(|x| x as u8).unwrap_or(0); data.push(v); }
        Some(Self{w,h,data})
    }
    
    pub fn sample(&self, u: f64, v: f64)->Vec3 {
        // wrap
        let uu = (u.fract()+1.0).fract();
        let vv = (v.fract()+1.0).fract();
        let x = ((uu * self.w as f64) as usize).min(self.w-1);
        let y = (( (1.0 - vv) * self.h as f64) as usize).min(self.h-1);
        let idx = (y*self.w + x)*3;
        Vec3::new(self.data[idx] as f64/255.0, self.data[idx+1] as f64/255.0, self.data[idx+2] as f64/255.0)
    }
}
