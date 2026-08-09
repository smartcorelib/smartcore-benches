// Deterministic instruction-count benchmark for `CoverTree::new` + `find`.
// Linux-only (requires Valgrind). The criterion analogue is
// `benches/cover_tree.rs`. A single mid-size grid (10k×10) is used: iai runs
// under Valgrind, and this is enough to regress both build and query paths
// without spending 100k-point Valgrind minutes.
#[cfg(target_os = "linux")]
use iai_callgrind::{library_benchmark, library_benchmark_group, main};
#[cfg(target_os = "linux")]
use smartcore::algorithm::neighbour::cover_tree::CoverTree;
#[cfg(target_os = "linux")]
use smartcore::metrics::distance::euclidian::Euclidian;
#[cfg(target_os = "linux")]
use smartcore::numbers::realnum::RealNumber;

#[cfg(target_os = "linux")]
fn make_data(n_samples: usize, n_features: usize) -> Vec<Vec<f64>> {
    (0..n_samples)
        .map(|_| (0..n_features).map(|_| f64::rand()).collect::<Vec<f64>>())
        .collect()
}

#[cfg(target_os = "linux")]
#[library_benchmark]
fn bench_cover_tree_build_10k_x_10() {
    let data = make_data(10_000, 10);
    let tree = CoverTree::new(data, Euclidian::<f64>::new()).unwrap();
    std::hint::black_box(tree);
}

#[cfg(target_os = "linux")]
#[library_benchmark]
fn bench_cover_tree_find_10k_x_10() {
    let data = make_data(10_000, 10);
    let query = data[0].clone();
    let tree = CoverTree::new(data, Euclidian::<f64>::new()).unwrap();
    let neighbors = tree.find(&query, 10).unwrap();
    std::hint::black_box(neighbors.len());
}

#[cfg(target_os = "linux")]
library_benchmark_group!(
    name = cover_tree;
    benchmarks = bench_cover_tree_build_10k_x_10, bench_cover_tree_find_10k_x_10
);

#[cfg(target_os = "linux")]
main!(library_benchmark_groups = cover_tree);

#[cfg(not(target_os = "linux"))]
fn main() {
    eprintln!("iai-callgrind benches are Linux-only (Valgrind). Skipping on this platform.");
}
