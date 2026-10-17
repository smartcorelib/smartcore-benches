#[cfg(target_os = "linux")]
use iai_callgrind::{library_benchmark, library_benchmark_group, main};
#[cfg(target_os = "linux")]
use smartcore::linalg::basic::matrix::DenseMatrix;
#[cfg(target_os = "linux")]
use smartcore::tree::decision_tree_classifier::{
    DecisionTreeClassifier, DecisionTreeClassifierParameters,
};

#[cfg(target_os = "linux")]
#[library_benchmark]
fn bench_decision_tree_classifier_fit_200x10() {
    let values: Vec<f64> = (0..2000_u64)
        .map(|i| ((i * 2_654_435_761) % 1000) as f64 / 1000.0)
        .collect();
    let x = DenseMatrix::new(200, 10, values, false).unwrap();
    let y: Vec<u32> = (0..200).map(|i| (i % 3) as u32).collect();
    let params = DecisionTreeClassifierParameters::default();
    let model = DecisionTreeClassifier::fit(&x, &y, params).unwrap();
    std::hint::black_box(model.predict(&x).unwrap().len());
}

#[cfg(target_os = "linux")]
library_benchmark_group!(
    name = decision_tree_classifier;
    benchmarks = bench_decision_tree_classifier_fit_200x10
);

#[cfg(target_os = "linux")]
main!(library_benchmark_groups = decision_tree_classifier);

#[cfg(not(target_os = "linux"))]
fn main() {
    eprintln!("iai-callgrind benches are Linux-only (Valgrind). Skipping on this platform.");
}
