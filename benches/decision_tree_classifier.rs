use criterion::BenchmarkId;
use criterion::{Criterion, criterion_group, criterion_main};
use std::hint::black_box;

use smartcore::tree::decision_tree_classifier::{
    DecisionTreeClassifier, DecisionTreeClassifierParameters, SplitCriterion,
};
use smartcore_benches::class_data;

fn params(criterion: SplitCriterion) -> DecisionTreeClassifierParameters {
    DecisionTreeClassifierParameters::default().with_criterion(criterion)
}

const SHAPES: [(usize, usize); 3] = [(100, 10), (1000, 10), (1000, 50)];

fn criteria() -> [SplitCriterion; 3] {
    [
        SplitCriterion::Gini,
        SplitCriterion::Entropy,
        SplitCriterion::ClassificationError,
    ]
}

fn decision_tree_classifier_fit_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("DecisionTreeClassifier::fit");

    for (n_samples, n_features) in SHAPES {
        let (x, y) = class_data(n_samples, n_features);
        for criterion in criteria() {
            group.bench_with_input(
                BenchmarkId::from_parameter(format!(
                    "n_samples: {}, n_features: {}, criterion: {:?}",
                    n_samples, n_features, criterion
                )),
                &criterion,
                |b, criterion| {
                    b.iter(|| {
                        DecisionTreeClassifier::fit(
                            black_box(&x),
                            black_box(&y),
                            params(criterion.clone()),
                        )
                        .unwrap();
                    })
                },
            );
        }
    }
    group.finish();
}

fn decision_tree_classifier_predict_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("DecisionTreeClassifier::predict");

    for (n_samples, n_features) in SHAPES {
        let (x, y) = class_data(n_samples, n_features);
        for criterion in criteria() {
            let label = format!(
                "n_samples: {}, n_features: {}, criterion: {:?}",
                n_samples, n_features, criterion
            );
            let model = DecisionTreeClassifier::fit(&x, &y, params(criterion)).unwrap();
            group.bench_function(BenchmarkId::from_parameter(label), |b| {
                b.iter(|| {
                    model.predict(black_box(&x)).unwrap();
                })
            });
        }
    }
    group.finish();
}

criterion_group!(
    benches,
    decision_tree_classifier_fit_benchmark,
    decision_tree_classifier_predict_benchmark
);
criterion_main!(benches);
