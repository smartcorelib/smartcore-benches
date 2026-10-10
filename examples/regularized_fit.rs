use smartcore::linear::elastic_net::ElasticNet;
use smartcore::linear::lasso::Lasso;
use smartcore_benches::{dense_data, elastic_net_parameters, lasso_parameters};
use std::error::Error;
use std::hint::black_box;

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() != 4
        || !matches!(args[0].as_str(), "lasso" | "elastic_net")
        || !matches!(args[3].as_str(), "row" | "column")
    {
        return Err("usage: regularized_fit <lasso|elastic_net> <rows> <cols> <row|column>".into());
    }
    let rows: usize = args[1].parse()?;
    let cols: usize = args[2].parse()?;
    if rows <= cols || cols == 0 {
        return Err("rows must exceed cols, and cols must be positive".into());
    }
    let (x, y) = dense_data(rows, cols, args[3] == "column");
    match args[0].as_str() {
        "lasso" => {
            black_box(Lasso::fit(&x, &y, lasso_parameters(true))?);
        }
        "elastic_net" => {
            black_box(ElasticNet::fit(&x, &y, elastic_net_parameters(true))?);
        }
        _ => unreachable!(),
    }
    Ok(())
}
