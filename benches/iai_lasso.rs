#[cfg(target_os = "linux")]
use iai_callgrind::{library_benchmark, library_benchmark_group};
#[cfg(target_os = "linux")]
use smartcore::linalg::basic::matrix::DenseMatrix;
#[cfg(target_os = "linux")]
use smartcore::linear::lasso::Lasso;
#[cfg(target_os = "linux")]
use smartcore_benches::{dense_data, lasso_parameters};

#[cfg(target_os = "linux")]
fn setup(column_major: bool, normalize: bool) -> (DenseMatrix<f64>, Vec<f64>, bool) {
    let (x, y) = dense_data(1024, 32, column_major);
    (x, y, normalize)
}

#[cfg(target_os = "linux")]
#[library_benchmark]
#[bench::fit(args = (false, true), setup = setup)]
fn row_major((x, y, normalize): (DenseMatrix<f64>, Vec<f64>, bool)) {
    std::hint::black_box(Lasso::fit(&x, &y, lasso_parameters(normalize)).unwrap());
}

#[cfg(target_os = "linux")]
#[library_benchmark]
#[bench::fit(args = (true, true), setup = setup)]
fn column_major((x, y, normalize): (DenseMatrix<f64>, Vec<f64>, bool)) {
    std::hint::black_box(Lasso::fit(&x, &y, lasso_parameters(normalize)).unwrap());
}

#[cfg(target_os = "linux")]
#[library_benchmark]
#[bench::fit(args = (false, false), setup = setup)]
fn without_normalization((x, y, normalize): (DenseMatrix<f64>, Vec<f64>, bool)) {
    std::hint::black_box(Lasso::fit(&x, &y, lasso_parameters(normalize)).unwrap());
}

#[cfg(target_os = "linux")]
library_benchmark_group!(name = lasso; benchmarks = row_major, column_major, without_normalization);

#[cfg(target_os = "linux")]
iai_callgrind::main!(library_benchmark_groups = lasso);

#[cfg(not(target_os = "linux"))]
fn main() {
    eprintln!("iai-callgrind benches require Linux and Valgrind.");
}
