use criterion::BenchmarkId;
use criterion::{black_box, criterion_group, criterion_main, Criterion};

use smartcore::linalg::basic::arrays::Array2 as BaseArray2;
use smartcore::linalg::basic::matrix::DenseMatrix;
use smartcore::linear::linear_regression::LinearRegression;

pub fn linear_regression_fit_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("LinearRegression::fit");

    for n_samples in [10_usize, 20_usize, 100_usize].iter() {
        for n_features in [10_usize, 100_usize, 1000_usize].iter() {
            let x = DenseMatrix::<f64>::rand(*n_samples, *n_features);
            let y: Vec<usize> = (0..*n_samples)
                .map(|i| (i % *n_samples / 5_usize) as usize)
                .collect::<Vec<usize>>();
            group.bench_with_input(
                BenchmarkId::from_parameter(format!(
                    "n_samples: {}, n_features: {}",
                    n_samples, n_features
                )),
                n_samples,
                |b, _| {
                    b.iter(|| {
                        LinearRegression::fit(black_box(&x), black_box(&y), Default::default()).unwrap();
                    })
                },
            );
        }
    }
    group.finish();
}

criterion_group!(
    benches,
    linear_regression_fit_benchmark,
);
criterion_main!(benches);
