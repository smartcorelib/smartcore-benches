use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use smartcore::linalg::basic::arrays::Array2;
use smartcore::linalg::basic::matrix::DenseMatrix;
use std::hint::black_box;

/// Benchmarks `Array2::matmul` directly, isolating the hot path at
/// `src/linalg/basic/arrays.rs:1117`. Sizes span the {64, 256, 1024}² grid
/// from the issue. Inputs use `RealNumber::rand()` (seed-0 deterministic when
/// smartcore is built without `std_rand`) so criterion and iai-callgrind see
/// identical inputs across runs.
fn matmul_bench(c: &mut Criterion) {
    let mut group = c.benchmark_group("DenseMatrix::matmul");

    for n in [64_usize, 256, 1024] {
        // Owned outside the timed region so allocation cost stays out of the
        // measurement; only the matmul is timed.
        let a = DenseMatrix::<f64>::rand(n, n);
        let b = DenseMatrix::<f64>::rand(n, n);

        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |bencher, _| {
            bencher.iter(|| {
                // Bind the result: criterion's closure tail is the timed
                // expression; `matmul` owns its output, so black_box the
                // returned matrix to prevent the optimizer from eliding it.
                let c = a.matmul(black_box(&b));
                black_box(c);
            });
        });
    }

    group.finish();
}

criterion_group!(benches, matmul_bench);
criterion_main!(benches);
