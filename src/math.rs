use ndarray::Array1;

/// Computes the Z-Score Distribution Normalization for a set of values.
///
/// Under Section 4.2 ("Two-Phase Retrieval"), MemRL addresses the incompatibility between
/// cosine similarity scores (bounded strictly between $[-1, 1]$) and learned Q-Value utilities
/// (which are theoretically unbounded scalars).
///
/// To fairly combine them in the Phase B Value-Aware Selection equation, both arrays
/// are passed through standard Z-score normalization:
/// $$\hat{x} = \frac{x - \mu}{\sigma}$$
///
/// This provides zero-mean and unit variance, making $\hat{\text{sim}}$ and $\hat{Q}$ directly comparable.
pub fn z_score_normalize(values: &[f32]) -> Vec<f32> {
    if values.is_empty() {
        return vec![];
    }
    if values.len() == 1 {
        return vec![0.0];
    }

    let arr = Array1::from_vec(values.to_vec());
    let mean = arr.mean().unwrap_or(0.0);
    let std = arr.std(0.0);

    if std == 0.0 {
        return vec![0.0; values.len()];
    }

    arr.mapv(|x| (x - mean) / std).to_vec()
}

/// Computes the composite Value-Aware Selection Score (Algorithm 1, Equation 7).
///
/// After Phase A retrieves a candidate pool of size $k_1$ based on semantic similarity,
/// Phase B re-ranks them to select the top $k_2$ items using a parameterized balance factor $\lambda$.
///
/// The formula is defined as:
/// $$\text{score}(s, z_i, e_i) = (1 - \lambda) \cdot \hat{\text{sim}}(\text{Emb}(s), \text{Emb}(z_i)) + \lambda \cdot \hat{Q}_i$$
///
/// - **$\lambda \in [0, 1]$**: The balancing scalar. $\lambda=0$ degenerates to pure semantic search (RAG).
///   $\lambda=1$ blindly trusts historical utility ignoring runtime context.
pub fn compute_memrl_score(sim_hat: f32, q_hat: f32, lambda: f32) -> f32 {
    (1.0 - lambda) * sim_hat + lambda * q_hat
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_z_score_normalization() {
        let values = vec![1.0, 2.0, 3.0];
        let normalized = z_score_normalize(&values);
        // mean is 2.0, std is ~0.8164966
        // check that 2.0 becomes 0.0
        assert!(normalized[0] < 0.0);
        assert!((normalized[1] - 0.0).abs() < 1e-5);
        assert!(normalized[2] > 0.0);
    }

    #[test]
    fn test_z_score_empty_or_single() {
        assert_eq!(z_score_normalize(&[]), Vec::<f32>::new());
        assert_eq!(z_score_normalize(&[4.0]), vec![0.0]);
    }

    #[test]
    fn test_compute_memrl_score() {
        assert_eq!(compute_memrl_score(0.8, -0.5, 0.0), 0.8); // pure sim
        assert_eq!(compute_memrl_score(-1.0, 1.5, 1.0), 1.5); // pure Q-value
        let val = compute_memrl_score(1.0, -1.0, 0.5); // 0.5 * 1.0 + 0.5 * -1.0
        assert!((val - 0.0).abs() < 1e-5);
    }
}
