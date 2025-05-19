use anyhow::Result;
use hmmm::HMM;
use ndarray::Array1;
use rand::thread_rng;

/// Trains an HMM using Baum-Welch and then finds the most likely sequence of hidden states
/// using the Viterbi algorithm.
///
/// # Arguments
/// * `observations` - A slice of u8 representing the quantized observation sequence.
/// * `num_states` - The number of hidden states for the HMM (N).
/// * `quant_max_value` - The inclusive maximum value of the quantized observations (e.g., if 0-3, then 3).
///                       This is used to determine K, the number of distinct observation symbols (K = quant_max_value + 1).
/// * `_num_iterations` - Currently unused by the hmmm::HMM::train method, which has its own convergence criteria.
///
/// # Returns
/// A `Result` containing a `Vec<usize>` of the most likely hidden states, or an error.
pub fn viterbi_path_from_observations(
    observations: &[u8],
    num_states: usize,
    quant_max_value: u8,
    _num_iterations: usize, // Kept for API consistency, though not used by hmmm::HMM::train
) -> Result<Vec<usize>> {
    log::info!(
        "Attempting Viterbi path calculation: num_states={}, quant_max_value={}, observation_len={}",
        num_states,
        quant_max_value,
        observations.len()
    );
    log::warn!("The '_num_iterations' parameter is currently not used by the hmmm::HMM::train function.");

    if observations.is_empty() {
        log::warn!("Observation sequence is empty. Returning empty Viterbi path.");
        return Ok(Vec::new());
    }
    if num_states == 0 {
        anyhow::bail!("Number of HMM states (num_states) cannot be zero.");
    }

    // Convert observations from &[u8] to ndarray::Array1<usize>
    let obs_array: Array1<usize> =
        Array1::from_iter(observations.iter().map(|&x| x as usize));

    // K: Number of distinct observation symbols.
    // If quant_max_value is, e.g., 3 (meaning observations can be 0, 1, 2, 3),
    // then K is 4.
    let k_observation_symbols = quant_max_value as usize + 1;

    if k_observation_symbols == 0 {
        anyhow::bail!("Number of observation symbols (k_observation_symbols derived from quant_max_value) cannot be zero.");
    }

    // Initialize a random number generator for HMM::train
    let mut rng = thread_rng();

    // Train the HMM using Baum-Welch
    // HMM::train takes: observation sequence, N (num_states), K (num_observation_symbols), rng
    log::debug!("Training HMM (Baum-Welch)... N={}, K={}", num_states, k_observation_symbols);
    let trained_hmm = HMM::train(&obs_array, num_states, k_observation_symbols, &mut rng);
    log::info!("HMM training complete.");

    // Get the most likely sequence of states using Viterbi
    log::debug!("Running Viterbi algorithm...");
    let state_sequence_array = trained_hmm.most_likely_sequence(&obs_array);
    let state_sequence_vec = state_sequence_array.into_raw_vec();
    log::info!("Viterbi path calculated. Path length: {}. First 5 states (if available): {:?}", 
        state_sequence_vec.len(), 
        state_sequence_vec.iter().take(5).collect::<Vec<_>>()
    );
    log::info!("Output summary: Viterbi path length={}", state_sequence_vec.len());
    Ok(state_sequence_vec)
}
