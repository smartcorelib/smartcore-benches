use criterion::BenchmarkId;
use criterion::{Criterion, criterion_group, criterion_main};
use std::hint::black_box;

use smartcore::ensemble::random_forest_classifier::{
    RandomForestClassifier, RandomForestClassifierParameters,
};
use smartcore_benches::class_data;

fn params(n_trees: u16) -> RandomForestClassifierParameters {
    RandomForestClassifierParameters::default()
        .with_n_trees(n_trees)
        .with_seed(42)
}

const SHAPES: [(usize, usize); 3] = [(100, 10), (1000, 10), (1000, 50)];
const N_TREES: [u16; 3] = [10, 50, 100];

fn random_forest_classifier_fit_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("RandomForestClassifier::fit");

    for (n_samples, n_features) in SHAPES {
        let (x, y) = class_data(n_samples, n_features);
        for n_trees in N_TREES {
            group.bench_with_input(
                BenchmarkId::from_parameter(format!(
                    "n_samples: {}, n_features: {}, n_trees: {}",
                    n_samples, n_features, n_trees
                )),
                &n_trees,
                |b, &n_trees| {
                    b.iter(|| {
                        RandomForestClassifier::fit(black_box(&x), black_box(&y), params(n_trees))
                            .unwrap();
                    })
                },
            );
        }
    }
    group.finish();
}

fn random_forest_classifier_predict_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("RandomForestClassifier::predict");

    for (n_samples, n_features) in SHAPES {
        let (x, y) = class_data(n_samples, n_features);
        for n_trees in N_TREES {
            let model = RandomForestClassifier::fit(&x, &y, params(n_trees)).unwrap();
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
    random_forest_classifier_fit_benchmark,
    random_forest_classifier_predict_benchmark
);
criterion_main!(benches);
