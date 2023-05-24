use opensrdk_linear_algebra::Matrix;
use opensrdk_symbolic_computation::{ConstantValue, Expression};

#[test]
fn test_main() {
    let n = 10;
    let zero = Matrix::from(1, vec![0.0; n]).unwrap();

    let mut wv = Expression::from(zero);

    let mut wq = Expression::Constant(ConstantValue::Scalar(0.0));
    let alpha = Expression::Constant(ConstantValue::Scalar(0.1));
    let r = Expression::Constant(ConstantValue::Scalar(0.1));
    let delta = r + alpha;
}

fn Q(V_hat: Expression, f: Expression) -> Expression {
    V_hat + f
}

fn psi(
    z_i: Expression,
    h_i: Expression,
    theta_i: Expression,
    theta_others: Expression,
) -> Expression {
    let a_i = z_i + h_i * theta_i + theta_others;
    let log_a_i = a_i.ln();
    let psi = log_a_i.differential(&["theta_i"]);
    psi
}

fn update(wv: Expression, wq: Expression, alpha: Expression) {}
