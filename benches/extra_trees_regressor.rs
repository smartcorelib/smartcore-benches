use criterion::BenchmarkId;
use criterion::{Criterion, criterion_group, criterion_main};
use std::hint::black_box;

use smartcore::ensemble::extra_trees_regressor::{
    ExtraTreesRegressor, ExtraTreesRegressorParameters,
};
use smartcore_benches::dense_data;

fn params(n_trees: usize) -> ExtraTreesRegressorParameters {
    ExtraTreesRegressorParameters::default()
        .with_n_trees(n_trees)
        .with_seed(42)
}

const SHAPES: [(usize, usize); 3] = [(100, 10), (1000, 10), (1000, 50)];
const N_TREES: [usize; 3] = [10, 50, 100];

const M_SHAPE: (usize, usize) = (1000, 50);
const M_N_TREES: usize = 50;
const M_VALUES: [usize; 4] = [1, 7, 25, 50];

fn extra_trees_regressor_fit_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("ExtraTreesRegressor::fit");

    for (n_samples, n_features) in SHAPES {
        let (x, y) = dense_data(n_samples, n_features, false);
        for n_trees in N_TREES {
            group.bench_with_input(
                BenchmarkId::from_parameter(format!(
                    "n_samples: {}, n_features: {}, n_trees: {}",
                    n_samples, n_features, n_trees
                )),
                &n_trees,
                |b, &n_trees| {
                    b.iter(|| {
                        ExtraTreesRegressor::fit(black_box(&x), black_box(&y), params(n_trees))
                            .unwrap();
                    })
                },
            );
        }
    }
    group.finish();
}

fn extra_trees_regressor_predict_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("ExtraTreesRegressor::predict");

    for (n_samples, n_features) in SHAPES {
        let (x, y) = dense_data(n_samples, n_features, false);
        for n_trees in N_TREES {
            let model = ExtraTreesRegressor::fit(&x, &y, params(n_trees)).unwrap();
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

fn extra_trees_regressor_fit_m_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("ExtraTreesRegressor::fit (m)");

    let (n_samples, n_features) = M_SHAPE;
    let (x, y) = dense_data(n_samples, n_features, false);
    for m in M_VALUES {
        group.bench_with_input(
            BenchmarkId::from_parameter(format!(
                "n_samples: {}, n_features: {}, n_trees: {}, m: {}",
                n_samples, n_features, M_N_TREES, m
            )),
            &m,
            |b, &m| {
                b.iter(|| {
                    ExtraTreesRegressor::fit(
                        black_box(&x),
                        black_box(&y),
                        params(M_N_TREES).with_m(m),
                    )
                    .unwrap();
                })
            },
        );
    }
    group.finish();
}

fn extra_trees_regressor_predict_m_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("ExtraTreesRegressor::predict (m)");

    let (n_samples, n_features) = M_SHAPE;
    let (x, y) = dense_data(n_samples, n_features, false);
    for m in M_VALUES {
        let model = ExtraTreesRegressor::fit(&x, &y, params(M_N_TREES).with_m(m)).unwrap();
        group.bench_with_input(
            BenchmarkId::from_parameter(format!(
                "n_samples: {}, n_features: {}, n_trees: {}, m: {}",
                n_samples, n_features, M_N_TREES, m
            )),
            &m,
            |b, _| {
                b.iter(|| {
                    model.predict(black_box(&x)).unwrap();
                })
            },
        );
    }
    group.finish();
}

criterion_group!(
    benches,
    extra_trees_regressor_fit_benchmark,
    extra_trees_regressor_predict_benchmark,
    extra_trees_regressor_fit_m_benchmark,
    extra_trees_regressor_predict_m_benchmark
);
criterion_main!(benches);
