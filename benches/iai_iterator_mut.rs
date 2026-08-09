// Deterministic instruction-count benchmark for the `iterator_mut` hot paths.
// Linux-only (requires Valgrind). The criterion analogue is
// `benches/iterator_mut.rs`.
//
// This is the gate that makes the #368 unsafe→safe `split_at_mut` refactor
// machine-verifiable: the fast path (axis matches storage order) should be
// ~0% instruction-count delta; the cross-axis path is where the refactor
// changes the work. A 1024² grid is enough to surface the change without
// spending 4096² Valgrind minutes; counts scale linearly.
#[cfg(target_os = "linux")]
use iai_callgrind::{library_benchmark, library_benchmark_group, main};
#[cfg(target_os = "linux")]
use smartcore::linalg::basic::arrays::{Array, Array2, MutArray};
#[cfg(target_os = "linux")]
use smartcore::linalg::basic::matrix::DenseMatrix;

#[cfg(target_os = "linux")]
#[library_benchmark]
fn bench_dense_iterator_mut_fast_path() {
    // Row-major storage, axis 0: the fast path that shortcuts to a contiguous
    // `values.iter_mut()` traversal. The #368 refactor should leave this ~unchanged.
    let mut m = DenseMatrix::<f64>::rand(1024, 1024);
    let mut i = 0u64;
    m.iterator_mut(0).for_each(|v| {
        *v = i as f64;
        i += 1;
    });
    std::hint::black_box(i);
}

#[cfg(target_os = "linux")]
#[library_benchmark]
fn bench_dense_iterator_mut_cross_axis() {
    // Row-major storage, axis 1: the cross-axis path. This is the #368 surface
    // where the safe `split_at_mut` permute replaces the raw-pointer traversal.
    let mut m = DenseMatrix::<f64>::rand(1024, 1024);
    let mut i = 0u64;
    m.iterator_mut(1).for_each(|v| {
        *v = i as f64;
        i += 1;
    });
    std::hint::black_box(i);
}

#[cfg(target_os = "linux")]
#[library_benchmark]
fn bench_mutview_iterator_mut() {
    // Full-matrix mutable view (`slice_mut` → `MutArrayView2::iterator_mut`).
    let mut m = DenseMatrix::<f64>::rand(1024, 1024);
    let n = 1024;
    let mut view = m.slice_mut(0..n, 0..n);
    let mut i = 0u64;
    view.iterator_mut(0).for_each(|v| {
        *v = i as f64;
        i += 1;
    });
    std::hint::black_box(i);
}

#[cfg(target_os = "linux")]
library_benchmark_group!(
    name = iterator_mut;
    benchmarks = bench_dense_iterator_mut_fast_path,
        bench_dense_iterator_mut_cross_axis,
        bench_mutview_iterator_mut
);

#[cfg(target_os = "linux")]
main!(library_benchmark_groups = iterator_mut);

#[cfg(not(target_os = "linux"))]
fn main() {
    eprintln!("iai-callgrind benches are Linux-only (Valgrind). Skipping on this platform.");
}
