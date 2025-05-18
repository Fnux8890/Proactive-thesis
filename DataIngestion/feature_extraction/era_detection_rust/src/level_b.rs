use changepoint::{Bocpd, BocpdLike};
use changepoint::rv::prelude::NormalGamma;
use anyhow::Result;


pub fn bocpd_probabilities(signal: &[f64], expected_run_length: f64) -> Result<Vec<f64>> {
    if signal.is_empty() {
        return Ok(Vec::new());
    }
    let prior = NormalGamma::new_unchecked(0.0, 1.0, 1.0, 1.0);
    let mut bocpd_model = Bocpd::new(expected_run_length, prior);
    
    let probabilities: Vec<f64> = signal.iter().map(|&val| {
        let posterior_slice: &[f64] = bocpd_model.step(&val); // Corrected: pass &val
        *posterior_slice.last().unwrap_or(&0.0_f64)
    }).collect();
    Ok(probabilities)
}
