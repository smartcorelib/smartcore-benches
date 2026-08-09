// Deterministic instruction-count benchmark for `Array2::ab`.
// Linux-only (requires Valgrind). The criterion analogue is `benches/ab.rs`.
// All four transpose combinations are exercised; the (false,false) branch
// routes through `matmul` while the other three take the indexed-loop path.
#[cfg(target_os = "linux")]
use iai_callgrind::{library_benchmark, library_benchmark_group, main};
#[cfg(target_os = "linux")]
use smartcore::linalg::basic::arrays::Array2;
#[cfg(target_os = "linux")]
use smartcore::linalg::basic::matrix::DenseMatrix;

#[cfg(target_os = "linux")]
#[library_benchmark]
fn bench_ab_false_false() {
    let a = DenseMatrix::<f64>::rand(256, 256);
    let b = DenseMatrix::<f64>::rand(256, 256);
    let c = a.ab(false, &b, false);
    std::hint::black_box(c);
}

#[cfg(target_os = "linux")]
#[library_benchmark]
fn bench_ab_false_true() {
    let a = DenseMatrix::<f64>::rand(256, 256);
    let b = DenseMatrix::<f64>::rand(256, 256);
    let c = a.ab(false, &b, true);
    std::hint::black_box(c);
}

#[cfg(target_os = "linux")]
#[library_benchmark]
fn bench_ab_true_false() {
    let a = DenseMatrix::<f64>::rand(256, 256);
    let b = DenseMatrix::<f64>::rand(256, 256);
    let c = a.ab(true, &b, false);
    std::hint::black_box(c);
}

#[cfg(target_os = "linux")]
#[library_benchmark]
fn bench_ab_true_true() {
    let a = DenseMatrix::<f64>::rand(256, 256);
    let b = DenseMatrix::<f64>::rand(256, 256);
    let c = a.ab(true, &b, true);
    std::hint::black_box(c);
}

#[cfg(target_os = "linux")]
library_benchmark_group!(
    name = ab;
    benchmarks = bench_ab_false_false, bench_ab_false_true, bench_ab_true_false, bench_ab_true_true
);

#[cfg(target_os = "linux")]
main!(library_benchmark_groups = ab);

#[cfg(not(target_os = "linux"))]
fn main() {
    eprintln!("iai-callgrind benches are Linux-only (Valgrind). Skipping on this platform.");
}
