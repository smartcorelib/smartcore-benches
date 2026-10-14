use ndarray::Array2;
use smartcore::linalg::basic::matrix::DenseMatrix;
use smartcore::linear::elastic_net::ElasticNetParameters;
use smartcore::linear::lasso::LassoParameters;

pub const FIT_SIZES: [(usize, usize); 4] = [(1024, 32), (8192, 32), (65536, 32), (1024, 128)];

fn value(row: usize, col: usize) -> f64 {
    let mut bits = (row as u64)
        .wrapping_mul(0x9e3779b97f4a7c15)
        .wrapping_add(col as u64)
        .wrapping_add(42);
    bits = (bits ^ (bits >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    bits = (bits ^ (bits >> 27)).wrapping_mul(0x94d049bb133111eb);
    bits ^= bits >> 31;
    let uniform = (bits >> 11) as f64 / ((1_u64 << 53) as f64);
    (uniform - 0.5) * (1.0 + (col % 5) as f64) + (col % 3) as f64
}

fn targets(rows: usize, cols: usize) -> Vec<f64> {
    (0..rows)
        .map(|row| {
            2.0 + (0..cols.min(8))
                .map(|col| value(row, col) * if col % 2 == 0 { 1.0 } else { -0.5 })
                .sum::<f64>()
                + 0.05 * value(row, cols)
        })
        .collect()
}

pub fn dense_data(rows: usize, cols: usize, column_major: bool) -> (DenseMatrix<f64>, Vec<f64>) {
    let values = (0..rows * cols)
        .map(|index| {
            let (row, col) = if column_major {
                (index % rows, index / rows)
            } else {
                (index / cols, index % cols)
            };
            value(row, col)
        })
        .collect();
    (
        DenseMatrix::new(rows, cols, values, column_major).unwrap(),
        targets(rows, cols),
    )
}

pub fn ndarray_data(rows: usize, cols: usize) -> (Array2<f64>, Vec<f64>) {
    (
        Array2::from_shape_fn((rows, cols), |(row, col)| value(row, col)),
        targets(rows, cols),
    )
}

pub fn lasso_parameters(normalize: bool) -> LassoParameters {
    LassoParameters::default()
        .with_alpha(0.1)
        .with_normalize(normalize)
        .with_tol(1e-4)
        .with_max_iter(1000)
}

pub fn elastic_net_parameters(normalize: bool) -> ElasticNetParameters {
    ElasticNetParameters::default()
        .with_alpha(0.1)
        .with_l1_ratio(0.5)
        .with_normalize(normalize)
        .with_tol(1e-4)
        .with_max_iter(1000)
}
