use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use smartcore::linalg::basic::arrays::Array2;
use smartcore::linalg::basic::matrix::DenseMatrix;

/// Benchmarks `Array2::ab` — the reachable matmul-with-transpose hot path
/// (smartcore 0.6.x has no `impl HighOrderOperations for DenseMatrix`, so
/// `high_order.rs:21` is currently orphaned; calls resolve to
/// `arrays.rs:1143`). This is the surface callers actually hit when they
/// request a transposed product, and it allocates special-cased indexing
/// loops rather than a `transpose()` + `matmul` pair.
///
/// All four `(a_transpose, b_transpose)` combinations are reported as
/// separate bench ids; each takes a different branch inside `ab`.
fn ab_bench(c: &mut Criterion) {
    let mut group = c.benchmark_group("Array2::ab");

    let n = 256_usize;
    let a = DenseMatrix::<f64>::rand(n, n);
    let b = DenseMatrix::<f64>::rand(n, n);

    for (a_t, b_t) in [(false, false), (false, true), (true, false), (true, true)] {
        let label = format!("a_t={a_t},b_t={b_t}");
        group.bench_with_input(BenchmarkId::new(label, n), &n, |bencher, _| {
            bencher.iter(|| {
                let result = a.ab(black_box(a_t), black_box(&b), black_box(b_t));
                black_box(result);
            });
        });
    }

    group.finish();
}

criterion_group!(benches, ab_bench);
criterion_main!(benches);
