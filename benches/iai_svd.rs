// Deterministic instruction-count benchmark for `SVDDecomposable::svd`.
// Linux-only (requires Valgrind). The criterion analogue is `benches/svd.rs`.
// A single square shape is used: iai runs under Valgrind (slow), and the SVD
// algorithm's instruction count scales predictably enough that a 128x128
// regression reliably flags a 256x256 regression too.
#[cfg(target_os = "linux")]
use iai_callgrind::{library_benchmark, library_benchmark_group, main};
#[cfg(target_os = "linux")]
use smartcore::linalg::basic::arrays::Array2;
#[cfg(target_os = "linux")]
use smartcore::linalg::basic::matrix::DenseMatrix;
#[cfg(target_os = "linux")]
use smartcore::linalg::traits::svd::SVDDecomposable;

#[cfg(target_os = "linux")]
#[library_benchmark]
fn bench_svd_square_128() {
    let a = DenseMatrix::<f64>::rand(128, 128);
    let svd = a.svd().unwrap();
    std::hint::black_box(svd.s.len());
}

#[cfg(target_os = "linux")]
library_benchmark_group!(name = svd; benchmarks = bench_svd_square_128);

#[cfg(target_os = "linux")]
main!(library_benchmark_groups = svd);

#[cfg(not(target_os = "linux"))]
fn main() {
    eprintln!("iai-callgrind benches are Linux-only (Valgrind). Skipping on this platform.");
}
