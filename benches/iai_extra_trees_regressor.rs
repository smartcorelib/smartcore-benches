#[cfg(target_os = "linux")]
use iai_callgrind::{library_benchmark, library_benchmark_group, main};
#[cfg(target_os = "linux")]
use smartcore::ensemble::extra_trees_regressor::{
    ExtraTreesRegressor, ExtraTreesRegressorParameters,
};
#[cfg(target_os = "linux")]
use smartcore::linalg::basic::matrix::DenseMatrix;

#[cfg(target_os = "linux")]
#[library_benchmark]
fn bench_extra_trees_regressor_fit_200x10() {
    let values: Vec<f64> = (0..2000_u64)
        .map(|i| ((i * 2_654_435_761) % 1000) as f64 / 1000.0)
        .collect();
    let x = DenseMatrix::new(200, 10, values, false).unwrap();
    let y: Vec<f64> = (0..200).map(|i| (i % 17) as f64 * 0.5).collect();
    let params = ExtraTreesRegressorParameters::default()
        .with_n_trees(10)
        .with_seed(42);
    let model = ExtraTreesRegressor::fit(&x, &y, params).unwrap();
    std::hint::black_box(model.predict(&x).unwrap().len());
}

#[cfg(target_os = "linux")]
library_benchmark_group!(
    name = extra_trees_regressor;
    benchmarks = bench_extra_trees_regressor_fit_200x10
);

#[cfg(target_os = "linux")]
main!(library_benchmark_groups = extra_trees_regressor);

#[cfg(not(target_os = "linux"))]
fn main() {
    eprintln!("iai-callgrind benches are Linux-only (Valgrind). Skipping on this platform.");
}
