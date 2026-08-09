// Deterministic instruction-count benchmark for `Array2::matmul`.
// Linux-only (requires Valgrind). The criterion analogue is `benches/matmul.rs`.
// Sizes are smaller than the criterion grid since iai runs each input once
// under Valgrind (slow) — but instruction counts are machine-independent so
// the smaller sizes still catch a regression that would scale to 1024².
#[cfg(target_os = "linux")]
use iai_callgrind::{library_benchmark, library_benchmark_group, main};
#[cfg(target_os = "linux")]
use smartcore::linalg::basic::arrays::Array2;
#[cfg(target_os = "linux")]
use smartcore::linalg::basic::matrix::DenseMatrix;

#[cfg(target_os = "linux")]
fn make(n: usize) -> DenseMatrix<f64> {
    DenseMatrix::<f64>::rand(n, n)
}

#[cfg(target_os = "linux")]
#[library_benchmark]
fn bench_matmul_64() {
    let a = make(64);
    let b = make(64);
    let c = a.matmul(&b);
    std::hint::black_box(c);
}

#[cfg(target_os = "linux")]
#[library_benchmark]
fn bench_matmul_256() {
    let a = make(256);
    let b = make(256);
    let c = a.matmul(&b);
    std::hint::black_box(c);
}

#[cfg(target_os = "linux")]
library_benchmark_group!(name = matmul; benchmarks = bench_matmul_64, bench_matmul_256);

#[cfg(target_os = "linux")]
main!(library_benchmark_groups = matmul);

#[cfg(not(target_os = "linux"))]
fn main() {
    eprintln!("iai-callgrind benches are Linux-only (Valgrind). Skipping on this platform.");
}
