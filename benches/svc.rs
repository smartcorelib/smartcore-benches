use criterion::BenchmarkId;
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use smartcore::linalg::basic::arrays::Array2 as BaseArray2;
use smartcore::linalg::basic::matrix::DenseMatrix;
use smartcore::svm::svc::{MultiClassSVC, SVCParameters};
use smartcore::svm::Kernels;

pub fn multiclass_svc_fit_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("MultiClassSVC::fit");

    for n_samples in [100_usize, 1000_usize, 10000_usize].iter() {
        for n_features in [10_usize, 100_usize, 1000_usize].iter() {
            let x = DenseMatrix::<f64>::rand(*n_samples, *n_features);
            let y: Vec<usize> = (0..*n_samples)
                .map(|i| (i % *n_samples / 5_usize) as usize)
                .collect::<Vec<usize>>();
            let parameters = SVCParameters::default()
            .with_c(1.0)
            .with_kernel(Kernels::rbf().with_gamma(0.7));
            group.bench_with_input(
                BenchmarkId::from_parameter(format!(
                    "n_samples: {}, n_features: {}",
                    n_samples, n_features
                )),
                n_samples,
                |b, _| {
                    b.iter(|| {
                        MultiClassSVC::fit(black_box(&x), black_box(&y), &parameters).unwrap();
                    })
                },
            );
        }
    }
    group.finish();
}

criterion_group!(
    benches,
    multiclass_svc_fit_benchmark,
);
criterion_main!(benches);
