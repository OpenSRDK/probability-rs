use super::rbf;
use DistributionProduct;
use MultivariateNormal;
use opensrdk_linear_algebra::Matrix;
use opensrdk_linear_algebra::Vector;
use opensrdk_symbolic_computation::new_partial_variable;
use opensrdk_symbolic_computation::new_variable_tensor;
use opensrdk_symbolic_computation::ExpressionArray;
use opensrdk_symbolic_computation::Size;
use opensrdk_symbolic_computation::{new_variable, Expression};
use std::iter::once;

// #[test]
fn test_gp() {
    let n = 20usize;
    let xd = 4usize;
    let y = (0..n).map(|yi| yi as f64).collect::<Vec<_>>();
    let y_mean = y.iter().sum::<f64>() / y.len() as f64;
    let x = vec![vec![1.0; xd]; n];
    let sigma = new_variable("sigma".to_string());
    let param = new_variable("theta".to_string());

    let k = new_partial_variable(ExpressionArray::from_factory(vec![n, n], |index| {
        let i = index[0];
        let j = index[1];
        rbf(x[i].clone().into(), x[j].clone().into(), param.clone())
    }));

    let normal = MultivariateNormal::new(y.into(), y_mean.into(), k + sigma, n);
}

fn test_recurrent_gp() {
    let n = 20usize;
    let yd = 2usize;
    let ud = 3usize;
    let xd = 4usize;
    let y = (0..n)
        .flat_map(|yi| vec![yi as f64; yd].into_iter())
        .collect::<Vec<_>>();
    let y_mean = y.iter().map(|yij| yij).sum::<f64>() / y.len() as f64;
    let u = ExpressionArray::from_factory(vec![n, ud], |indices| {
        new_variable(format!("u_{{{}, {}}}", indices[0], indices[1]))
    });
    let x = vec![vec![1.0; xd]; n];
    let cy = new_variable_tensor("cy".to_string(), vec![Size::Many, Size::Many]);
    let sigma_y = new_variable("sigma_y".to_string());
    let theta_y = new_variable("theta_y".to_string());
    let cu = new_variable_tensor("cu".to_string(), vec![Size::Many, Size::Many]);
    let sigma_u = new_variable("sigma_u".to_string());
    let theta_u = new_variable("theta_u".to_string());

    let ky = new_partial_variable(ExpressionArray::from_factory(vec![n, n], |index| {
        let i = index[0];
        let j = index[1];
        rbf(
            new_partial_variable(ExpressionArray::from_factory(vec![1, ud], |indices| {
                u[&[i, indices[1]]].clone()
            })),
            new_partial_variable(ExpressionArray::from_factory(vec![1, ud], |indices| {
                u[&[j, indices[1]]].clone()
            })),
            theta_y.clone(),
        )
    }));
    let ku = new_partial_variable(ExpressionArray::from_factory(vec![n, n], |index| {
        let i = index[0];
        let j = index[1];
        rbf(
            new_partial_variable(ExpressionArray::from_factory(vec![1, ud + xd], |indices| {
                if indices[1] < ud {
                    u[&[i, indices[1]]].clone()
                } else {
                    x[i][index[1]].into()
                }
            })),
            new_partial_variable(ExpressionArray::from_factory(vec![1, ud + xd], |indices| {
                if indices[1] < ud {
                    u[&[j, indices[1]]].clone()
                } else {
                    x[j][index[1]].into()
                }
            })),
            theta_u.clone(),
        )
    }));

    let normal = MultivariateNormal::new(y.into(), y_mean.into(), (ky + sigma_y).direct(cy), yd)
        * MultivariateNormal::new(
            new_partial_variable(u),
            0.0.into(),
            (ku + sigma_u).direct(cu),
            n,
        );
}
