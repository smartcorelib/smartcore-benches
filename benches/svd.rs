use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use ndarray::Array2 as NdArray2;
use smartcore::linalg::basic::arrays::Array2;
use smartcore::linalg::basic::matrix::DenseMatrix;
use smartcore::linalg::traits::svd::SVDDecomposable;
use std::hint::black_box;

/// Benchmarks `SVDDecomposable::svd` isolating the hot path at
/// `src/linalg/traits/svd.rs:71`. Two shapes (square and tall) are exercised
/// on both `DenseMatrix` and the ndarray backend so backend regressions are
/// visible independently of the decomposition algorithm.
fn svd_bench(c: &mut Criterion) {
    let mut group = c.benchmark_group("DenseMatrix::svd");

    for (rows, cols, label) in [(128_usize, 128_usize, "square"), (256, 64, "tall")] {
        let a = DenseMatrix::<f64>::rand(rows, cols);
        group.bench_with_input(
            BenchmarkId::new(label, format!("{rows}x{cols}")),
            &label,
            |bencher, _| {
                bencher.iter(|| {
                    let svd = a.svd().unwrap();
                    // Touch the outputs so the decomposition cannot be
                    // optimized away; `s` is the canonical correctness check.
                    black_box(svd.s.len());
                });
            },
        );
    }

    group.finish();
}

fn svd_ndarray_bench(c: &mut Criterion) {
    let mut group = c.benchmark_group("ndarray::Array2::svd");

    for (rows, cols, label) in [(128_usize, 128_usize, "square"), (256, 64, "tall")] {
        // ndarray's Array2 rand comes from smartcore's Array2 trait impl
        // (gated by ndarray-bindings), not ndarray itself.
        let a = NdArray2::<f64>::rand(rows, cols);
        group.bench_with_input(
            BenchmarkId::new(label, format!("{rows}x{cols}")),
            &label,
            |bencher, _| {
                bencher.iter(|| {
                    let svd = a.svd().unwrap();
                    black_box(svd.s.len());
                });
            },
        );
    }

    group.finish();
}

criterion_group!(benches, svd_bench, svd_ndarray_bench);
criterion_main!(benches);
