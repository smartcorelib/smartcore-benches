use criterion::BenchmarkId;
use criterion::{Criterion, black_box, criterion_group, criterion_main};

use smartcore::linalg::basic::arrays::Array2 as BaseArray2;
use smartcore::linalg::basic::matrix::DenseMatrix;
use smartcore::linear::linear_regression::LinearRegression;

/// Benchmarks `LinearRegression::fit` with the default SVD solver.
/// Grids keep n_samples > n_features + 1: `fit` augments X with a ones
/// column, so fewer samples than that leave the system underdetermined —
/// SVD indexes past b (smartcore `SVD::solve`) and QR raises
/// `Matrix is rank deficient`.
fn linear_regression_fit_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("LinearRegression::fit");

    for (n_samples, n_features) in [(64_usize, 16_usize), (256, 64), (1024, 256)] {
        let x = DenseMatrix::<f64>::rand(n_samples, n_features);
        let y: Vec<usize> = (0..n_samples)
            .map(|i| i % n_samples / 5_usize)
            .collect::<Vec<usize>>();
        group.bench_with_input(
            BenchmarkId::from_parameter(format!(
                "n_samples: {}, n_features: {}",
                n_samples, n_features
            )),
            &n_samples,
            |b, _| {
                b.iter(|| {
                    LinearRegression::fit(black_box(&x), black_box(&y), Default::default())
                        .unwrap();
                })
            },
        );
    }
    group.finish();
}

criterion_group!(benches, linear_regression_fit_benchmark,);
criterion_main!(benches);
