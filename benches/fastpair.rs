use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use itertools::Itertools;

// to run this bench you have to change the declaraion in mod.rs ---> pub mod fastpair;
use smartcore::algorithm::neighbour::fastpair::FastPair;
use smartcore::linalg::basic::arrays::{Array2, Array};
use smartcore::linalg::basic::matrix::DenseMatrix;
use smartcore::metrics::distance::PairwiseDistance;
use std::time::Duration;

/// Utilities substitutes for benches
/// 

///
/// return sum of squared distancs
/// 
fn squared_distance(x: Vec<f64>, y: Vec<f64>) -> f64 {
    if x.len() != y.len() {
        panic!("Input vector sizes are different.");
    }

    let sum: f64 = x
        .iter()
        .zip(y.iter())
        .map(|(&a, &b)| {
            let r = a - b;
            r * r
        })
        .sum();

    sum
}


///
/// Brute force algorithm, used only for comparison and testing
///
pub fn closest_pair_brute(samples: &DenseMatrix<f64>, n_samples: usize
) -> PairwiseDistance<f64> {
    let mut closest_pair = PairwiseDistance {
        node: 0,
        neighbour: Option::None,
        distance: Some(f64::MAX),
    };
    for pair in (0..n_samples).combinations(2) {
        let d = squared_distance(
            samples.get_row(pair[0]).iterator(0).copied().collect::<Vec<f64>>(),
            samples.get_row(pair[1]).iterator(0).copied().collect::<Vec<f64>>(),
        );
        if d < closest_pair.distance.unwrap() {
            closest_pair.node = pair[0];
            closest_pair.neighbour = Some(pair[1]);
            closest_pair.distance = Some(d);
        }
    }
    closest_pair
}

fn closest_pair_bench(n: usize, m: usize) -> () {
    let x = DenseMatrix::<f64>::rand(n, m);
    let fastpair = FastPair::new(&x);
    let result = fastpair.unwrap();

    result.closest_pair();
}

fn closest_pair_brute_bench(n: usize, m: usize) -> () {
    let x = DenseMatrix::<f64>::rand(n, m);
    closest_pair_brute(&x, x.shape().0);
}

fn bench_fastpair(c: &mut Criterion) {
    let mut group = c.benchmark_group("FastPair");

    // with full samples size (100) the test will take too long
    group.significance_level(0.1).sample_size(30);
    // increase from default 5.0 secs
    group.measurement_time(Duration::from_secs(60));

    for n_samples in [20_usize, 10_usize].iter() {
        for n_features in [10_usize, 100_usize, 1000_usize].iter() {
            group.bench_with_input(
                BenchmarkId::from_parameter(format!(
                    "fastpair --- n_samples: {}, n_features: {}",
                    n_samples, n_features
                )),
                n_samples,
                |b, _| b.iter(|| closest_pair_bench(*n_samples, *n_features)),
            );
            group.bench_with_input(
                BenchmarkId::from_parameter(format!(
                    "brute --- n_samples: {}, n_features: {}",
                    n_samples, n_features
                )),
                n_samples,
                |b, _| b.iter(|| closest_pair_brute_bench(*n_samples, *n_features)),
            );
        }
    }
    group.finish();
}

criterion_group!(benches, bench_fastpair);
criterion_main!(benches);
