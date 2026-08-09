use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use smartcore::linalg::basic::arrays::{Array, Array2, MutArray};
use smartcore::linalg::basic::matrix::DenseMatrix;

/// Benchmarks the `iterator_mut` hot paths referenced by issue #407:
///
/// - `DenseMatrix::iterator_mut` (`src/linalg/basic/matrix.rs:545`) — the
///   `#368` unsafe→`split_at_mut` refactor surface for owned matrices.
/// - `DenseMatrixMutView::iter_mut` (via `Array2::slice_mut` →
///   `MutArrayView2::iterator_mut`, `matrix.rs:254`) — the view path that
///   still holds the remaining `unsafe` trailing the #368 cleanup.
///
/// For each storage order (row-major and column-major) both traversal axes
/// are measured: the axis matching storage order is the fast path
/// (shortcuts to `values.iter_mut()`); the cross axis is the permute path
/// whose instruction count the #368 refactor changed. Sizes {1024, 4096}²
/// match the issue.
fn dense_iterator_mut_bench(c: &mut Criterion) {
    let mut group = c.benchmark_group("DenseMatrix::iterator_mut");

    for n in [1024_usize, 4096] {
        for column_major in [false, true] {
            let storage = if column_major {
                "col-major"
            } else {
                "row-major"
            };
            for axis in [0_u8, 1] {
                // Owned outside the timed region; each iter borrows `m` mutably
                // and writes through the iterator so the work cannot be
                // optimized away. The matrix is re-randomized implicitly by the
                // write — values are irrelevant, only the traversal cost.
                let mut m = DenseMatrix::<f64>::rand(n, n);
                // `rand` builds row-major; rebuild in the desired storage via
                // `from_2d_array`-free reordering using `from_iterator` over the
                // existing values in the requested axis order.
                if column_major {
                    let values: Vec<f64> = m.iterator(0).copied().collect();
                    m = DenseMatrix::from_iterator(values.into_iter(), n, n, 1);
                }

                let label = format!("{storage},axis={axis}");
                group.bench_with_input(BenchmarkId::new(label, n), &axis, |bencher, &axis| {
                    bencher.iter(|| {
                        let mut i = 0u64;
                        m.iterator_mut(black_box(axis)).for_each(|v| {
                            *v = i as f64;
                            i += 1;
                        });
                        black_box(i);
                    });
                });
            }
        }
    }

    group.finish();
}

fn mutview_iterator_mut_bench(c: &mut Criterion) {
    let mut group = c.benchmark_group("DenseMatrixMutView::iterator_mut");

    for n in [1024_usize, 4096] {
        for axis in [0_u8, 1] {
            let mut m = DenseMatrix::<f64>::rand(n, n);

            let label = format!("axis={axis}");
            group.bench_with_input(BenchmarkId::new(label, n), &axis, |bencher, &axis| {
                bencher.iter(|| {
                    // Full-matrix mutable view: exercises the MutView path
                    // (`slice_mut` → `MutArrayView2::iterator_mut`).
                    let mut view = m.slice_mut(0..n, 0..n);
                    let mut i = 0u64;
                    view.iterator_mut(black_box(axis)).for_each(|v| {
                        *v = i as f64;
                        i += 1;
                    });
                    black_box(i);
                });
            });
        }
    }

    group.finish();
}

criterion_group!(
    benches,
    dense_iterator_mut_bench,
    mutview_iterator_mut_bench
);
criterion_main!(benches);
