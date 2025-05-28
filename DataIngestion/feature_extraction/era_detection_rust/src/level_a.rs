use changepoint::{Bocpd, BocpdLike};
use changepoint::rv::prelude::NormalGamma;
use anyhow::Result;
use log::info; // Import info macro

// just for rode rabbit
// Helper function to find changepoints from BOCPD probabilities
fn find_changepoints_from_bocpd_probs(probs: &[f64], threshold: f64) -> Vec<usize> {
    let mut changepoints = Vec::new();
    if probs.is_empty() {
        return changepoints;
    }
    // A simple thresholding method: a point is a changepoint if its P(cp) > threshold
    // More sophisticated methods exist (e.g., looking for peaks, comparing to run length posteriors)
    // The original Python code did (cp_prob > 0.5).cumsum() for eras,
    // which means a new era starts when cp_prob > 0.5.
    // So, an index t is a changepoint (start of new segment) if probs[t] > threshold.
    for (i, &p) in probs.iter().enumerate() {
        if p > threshold {
            changepoints.push(i); // index i is the first point of the new segment
        }
    }
    // Post-process: merge close changepoints or apply min_segment_length if needed
    // For now, this is a raw list of points where P(cp) exceeded threshold.
    // To mimic PELT's output (indices of *first new point* of a segment), this might need adjustment
    // or we can adjust the era labeling in main.rs. The current approach is fine for thresholding.
    changepoints
}

pub fn detect_changepoints_level_a(
    signal: &[f64],
    expected_run_length: f64, 
    min_segment_length: usize, // min_segment_length for Level A, might be different from BOCPD lambda
    bocpd_threshold: f64,      // Probability threshold to declare a changepoint
) -> Result<Vec<usize>> {
    if signal.is_empty() {
        return Err(anyhow::anyhow!("Level A signal cannot be empty."));
    }
    if signal.len() < min_segment_length * 2 {
        return Err(anyhow::anyhow!(
            "Level A signal is too short (length: {}) for min_segment_length: {}. Needs at least {}.",
            signal.len(),
            min_segment_length,
            min_segment_length * 2
        ));
    }

    info!("Running BOCPD for Level A segmentation with lambda: {}, threshold: {}", expected_run_length, bocpd_threshold);

    let prior = NormalGamma::new_unchecked(0.0, 1.0, 1.0, 1.0);
    let mut bocpd_model = Bocpd::new(expected_run_length, prior);
    
    let probabilities: Vec<f64> = signal.iter().map(|&val| {
        let posterior_slice: &[f64] = bocpd_model.step(&val); // Corrected: pass &val
        *posterior_slice.last().unwrap_or(&0.0_f64) 
    }).collect();

    let changepoint_indices = find_changepoints_from_bocpd_probs(&probabilities, bocpd_threshold);

    // Optional: Add logic to ensure min_segment_length between identified changepoints
    // This would involve iterating through changepoint_indices and merging/removing
    // if segments are too short. For now, returning raw thresholded points.
    if changepoint_indices.is_empty() && signal.len() >= min_segment_length {
        // If no changepoints detected but signal is long enough, treat as one segment.
        // The calling code in main.rs expects a list of *internal* breakpoints.
        // So, an empty vector here implies one whole segment.
        return Ok(Vec::new());
    }

    Ok(changepoint_indices)
}
