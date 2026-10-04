use criterion::BenchmarkId;
use criterion::{Criterion, criterion_group, criterion_main};
use std::hint::black_box;

use smartcore::ensemble::random_forest_regressor::{
    RandomForestRegressor, RandomForestRegressorParameters,
};
use smartcore::linalg::basic::arrays::Array2 as BaseArray2;
use smartcore::linalg::basic::matrix::DenseMatrix;

fn params(n_trees: usize) -> RandomForestRegressorParameters {
    RandomForestRegressorParameters::default()
        .with_n_trees(n_trees)
        .with_seed(42)
}

fn target(n_samples: usize) -> Vec<f64> {
    (0..n_samples).map(|i| (i % 17) as f64 * 0.5).collect()
}

const SHAPES: [(usize, usize); 3] = [(100, 10), (1000, 10), (1000, 50)];
const N_TREES: [usize; 3] = [10, 50, 100];

fn random_forest_regressor_fit_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("RandomForestRegressor::fit");

    for (n_samples, n_features) in SHAPES {
        let x = DenseMatrix::<f64>::rand(n_samples, n_features);
        let y = target(n_samples);
        for n_trees in N_TREES {
            group.bench_with_input(
                BenchmarkId::from_parameter(format!(
                    "n_samples: {}, n_features: {}, n_trees: {}",
                    n_samples, n_features, n_trees
                )),
                &n_trees,
                |b, &n_trees| {
                    b.iter(|| {
                        RandomForestRegressor::fit(black_box(&x), black_box(&y), params(n_trees))
                            .unwrap();
                    })
                },
            );
        }
    }
    group.finish();
}

fn random_forest_regressor_predict_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("RandomForestRegressor::predict");

    for (n_samples, n_features) in SHAPES {
        let x = DenseMatrix::<f64>::rand(n_samples, n_features);
        let y = target(n_samples);
        for n_trees in N_TREES {
            let model = RandomForestRegressor::fit(&x, &y, params(n_trees)).unwrap();
            group.bench_with_input(
                BenchmarkId::from_parameter(format!(
                    "n_samples: {}, n_features: {}, n_trees: {}",
                    n_samples, n_features, n_trees
                )),
                &n_trees,
                |b, _| {
                    b.iter(|| {
                        model.predict(black_box(&x)).unwrap();
                    })
                },
            );
        }
    }
    group.finish();
}

criterion_group!(
    benches,
    random_forest_regressor_fit_benchmark,
    random_forest_regressor_predict_benchmark
);
criterion_main!(benches);
