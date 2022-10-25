#[macro_use]
extern crate criterion;
extern crate smartcore;

use criterion::black_box;
use criterion::Criterion;
use smartcore::metrics::distance::*;

fn criterion_benchmark(c: &mut Criterion) {
    let a = vec![1., 2., 3.];

    c.bench_function("Euclidean Distance", move |b| {
        b.iter(|| Distances::euclidian().distance(black_box(&a), black_box(&a)))
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
