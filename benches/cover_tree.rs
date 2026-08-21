use criterion::{BatchSize, BenchmarkId, Criterion, criterion_group, criterion_main};
use smartcore::algorithm::neighbour::cover_tree::CoverTree;
use smartcore::algorithm::neighbour::linear_search::LinearKNNSearch;
use smartcore::metrics::distance::euclidian::Euclidian;
use smartcore::numbers::realnum::RealNumber;
use std::hint::black_box;

/// Generates `n_samples` points of `n_features` dimensions using
/// `RealNumber::rand()`, which is seed-0 deterministic when smartcore is
/// built without `std_rand` — so cover_tree and linear_search see identical
/// inputs across runs and across the two backends.
fn make_data(n_samples: usize, n_features: usize) -> Vec<Vec<f64>> {
    (0..n_samples)
        .map(|_| (0..n_features).map(|_| f64::rand()).collect::<Vec<f64>>())
        .collect()
}

/// Benchmarks `CoverTree::new` (build) and `CoverTree::find` (query) at
/// {1k,10k,100k} × {10,100}, isolating the hot path at
/// `src/algorithm/neighbour/cover_tree.rs:37`. The same data is also run
/// through `LinearKNNSearch` as the brute-force baseline.
fn cover_tree_build_bench(c: &mut Criterion) {
    let mut group = c.benchmark_group("CoverTree::new");

    for n_samples in [1_000_usize, 10_000, 100_000] {
        for n_features in [10_usize, 100] {
            // Owned outside the timed region; `iter_batched` clones it (untimed)
            // before each measured call since `CoverTree::new` moves the data.
            let data = make_data(n_samples, n_features);
            let id = format!("{n_samples}x{n_features}");
            group.bench_with_input(BenchmarkId::new(id, "build"), &data, |bencher, data| {
                bencher.iter_batched(
                    || data.clone(),
                    |batch| {
                        let tree = CoverTree::new(batch, Euclidian::<f64>::new()).unwrap();
                        black_box(tree);
                    },
                    BatchSize::SmallInput,
                );
            });
        }
    }

    group.finish();
}

fn cover_tree_find_bench(c: &mut Criterion) {
    let mut group = c.benchmark_group("CoverTree::find");

    for n_samples in [1_000_usize, 10_000, 100_000] {
        for n_features in [10_usize, 100] {
            let data = make_data(n_samples, n_features);
            // Tree is built once (untimed); only `find` is measured.
            let tree = CoverTree::new(data.clone(), Euclidian::<f64>::new()).unwrap();
            // Query an existing point so `find` exercises the real traversal,
            // not the no-match path.
            let query = data[0].clone();
            let id = format!("{n_samples}x{n_features}");
            group.bench_with_input(BenchmarkId::new(id, "k=10"), &query, |bencher, query| {
                bencher.iter(|| {
                    let neighbors = tree.find(black_box(query), 10).unwrap();
                    black_box(neighbors.len());
                });
            });
        }
    }

    group.finish();
}

fn linear_knn_find_bench(c: &mut Criterion) {
    let mut group = c.benchmark_group("LinearKNNSearch::find");

    for n_samples in [1_000_usize, 10_000, 100_000] {
        for n_features in [10_usize, 100] {
            let data = make_data(n_samples, n_features);
            let knn = LinearKNNSearch::new(data.clone(), Euclidian::<f64>::new()).unwrap();
            let query = data[0].clone();
            let id = format!("{n_samples}x{n_features}");
            group.bench_with_input(BenchmarkId::new(id, "k=10"), &query, |bencher, query| {
                bencher.iter(|| {
                    let neighbors = knn.find(black_box(query), 10).unwrap();
                    black_box(neighbors.len());
                });
            });
        }
    }

    group.finish();
}

criterion_group!(
    benches,
    cover_tree_build_bench,
    cover_tree_find_bench,
    linear_knn_find_bench
);
criterion_main!(benches);
