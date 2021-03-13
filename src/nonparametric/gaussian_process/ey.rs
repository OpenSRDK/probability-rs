use rayon::prelude::*;

pub fn ey(y: &[f64]) -> f64 {
  y.par_iter().sum::<f64>() / y.len() as f64
}

pub fn y_ey(y: &[f64], ey: f64) -> Vec<f64> {
  y.par_iter().map(|&yi| yi - ey).collect::<Vec<_>>()
}
