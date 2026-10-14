use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use smartcore::linear::lasso::Lasso;
use smartcore_benches::{FIT_SIZES, dense_data, lasso_parameters, ndarray_data};
use std::hint::black_box;

fn fit_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("Lasso::fit");
    for (rows, cols) in FIT_SIZES {
        for column_major in [false, true] {
            let (x, y) = dense_data(rows, cols, column_major);
            let layout = if column_major {
                "dense_column_major"
            } else {
                "dense_row_major"
            };
            group.bench_function(BenchmarkId::new(layout, format!("{rows}x{cols}")), |b| {
                b.iter(|| {
                    black_box(
                        Lasso::fit(black_box(&x), black_box(&y), lasso_parameters(true)).unwrap(),
                    )
                });
            });
        }
        let (x, y) = ndarray_data(rows, cols);
        group.bench_function(
            BenchmarkId::new("ndarray_row_major", format!("{rows}x{cols}")),
            |b| {
                b.iter(|| {
                    black_box(
                        Lasso::fit(black_box(&x), black_box(&y), lasso_parameters(true)).unwrap(),
                    )
                });
            },
        );
    }
    group.finish();
}

criterion_group!(benches, fit_benchmark);
criterion_main!(benches);
