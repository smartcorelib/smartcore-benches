//! Coefficient of Determination (R2)
//!
//! Coefficient of determination, denoted R2 is the proportion of the variance in the dependent variable that can be explained be explanatory (independent) variable(s).
//!
//! \\[R^2(y, \hat{y}) = 1 - \frac{\sum_{i=1}^{n}(y_i - \hat{y_i})^2}{\sum_{i=1}^{n}(y_i - \bar{y})^2} \\]
//!
//! where \\(\hat{y}\\) are predictions, \\(y\\) are true target values, \\(\bar{y}\\) is the mean of the observed data
//!
//! Example:
//!
//! ```
//! use smartcore::metrics::mean_absolute_error::MeanAbsoluteError;
//! let y_pred: Vec<f64> = vec![3., -0.5, 2., 7.];
//! let y_true: Vec<f64> = vec![2.5, 0.0, 2., 8.];
//!
//! let mse: f64 = MeanAbsoluteError {}.get_score(&y_pred, &y_true);
//! ```
//!
//! <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
//! <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::linalg::base::Array1;
use crate::num::Number;

/// Coefficient of Determination (R2)
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug)]
pub struct R2 {}

impl R2 {
    /// Computes R2 score
    /// * `y_true` - Ground truth (correct) target values.
    /// * `y_pred` - Estimated target values.
    pub fn get_score<T: Number, V: Array1<T>>(&self, y_true: &V, y_pred: &V) -> f64 {
        if y_true.shape() != y_pred.shape() {
            panic!(
                "The vector sizes don't match: {} != {}",
                y_true.shape(),
                y_pred.shape()
            );
        }

        let n = y_true.shape();

        let mut mean = y_true.mean();
        let mut ss_tot = 0f64;
        let mut ss_res = 0f64;

        for i in 0..n {
            let y_i = y_true.get(i).to_f64().unwrap();
            let f_i = y_pred.get(i).to_f64().unwrap();
            ss_tot += (y_i - mean) * (y_i - mean);
            ss_res += (y_i - f_i) * (y_i - f_i);
        }

        1f64 - (ss_res / ss_tot)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
    #[test]
    fn r2() {
        let y_true: Vec<f64> = vec![3., -0.5, 2., 7.];
        let y_pred: Vec<f64> = vec![2.5, 0.0, 2., 8.];

        let score1: f64 = R2 {}.get_score(&y_true, &y_pred);
        let score2: f64 = R2 {}.get_score(&y_true, &y_true);

        assert!((score1 - 0.948608137).abs() < 1e-8);
        assert!((score2 - 1.0).abs() < 1e-8);
    }
}
