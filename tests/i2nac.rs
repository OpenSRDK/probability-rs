use std::{f32::consts::E, time::Instant};

use opensrdk_linear_algebra::Matrix;
use opensrdk_symbolic_computation::{ConstantValue, Expression, MatrixExpression};

#[test]
fn test_main() {
    let n = 10;
    let zero: Matrix = Matrix::from(1, vec![0.0; n]).unwrap();

    let mut wv = Expression::from(zero.clone());

    let mut wq = Expression::Constant(ConstantValue::Scalar(0.0));
    let alpha = Expression::Constant(ConstantValue::Scalar(0.1));
    let r = Expression::Constant(ConstantValue::Scalar(0.1));
    let delta = Expression::Constant(ConstantValue::Scalar(0.1));
}

fn update(wv: Expression, wq: Expression, alpha: Expression) {}
