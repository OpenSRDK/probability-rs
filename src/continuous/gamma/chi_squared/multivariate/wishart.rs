use crate::{DependentJoint, Distribution, IndependentJoint, RandomVariable};
use crate::{DistributionError, WishartParams};
use crate::{ExactMultivariateNormalParams, MultivariateNormal};
use opensrdk_linear_algebra::{matrix::ge::sy_he::po::trf::POTRF, *};
use rand::prelude::*;
use std::{ops::BitAnd, ops::Mul};

/// Wishart distribution
#[derive(Clone, Debug)]
pub struct Wishart;

#[derive(thiserror::Error, Debug)]
pub enum WishartError {
    #[error("Dimension mismatch")]
    DimensionMismatch,
    #[error("'n' must be >= dimension")]
    NMustBeGTEDimension,
}

impl Distribution for Wishart {
    type Value = Matrix;
    type Condition = WishartParams;

    /// x must be cholesky decomposed
    fn fk(&self, x: &Self::Value, theta: &Self::Condition) -> Result<f64, DistributionError> {
        let lv = theta.lv();
        let n = theta.n();

        let p = x.rows() as f64;

        Ok(x.trdet().powf((n + p + 1.0) / 2.0)
            * (-0.5 * POTRF(lv.clone()).potrs(x * x.t())?.tr()).exp())
    }

    /// output is cholesky decomposed
    fn sample(
        &self,
        theta: &Self::Condition,
        rng: &mut dyn RngCore,
    ) -> Result<Self::Value, DistributionError> {
        let lv = theta.lv();
        let n = theta.n() as usize;

        let p = lv.rows();

        let normal = MultivariateNormal::new();
        let normal_params = ExactMultivariateNormalParams::new(vec![0.0; p], lv.clone())?;

        let w = (0..n)
            .into_iter()
            .map(|_| normal.sample(&normal_params, rng))
            .try_fold::<Matrix, _, Result<Matrix, DistributionError>>(
                Matrix::new(p, p),
                |acc, v: Result<Vec<f64>, DistributionError>| {
                    let v = v?;
                    Ok(acc + v.clone().row_mat() * v.col_mat())
                },
            )?;

        Ok(w)
    }
}

impl<Rhs, TRhs> Mul<Rhs> for Wishart
where
    Rhs: Distribution<Value = TRhs, Condition = WishartParams>,
    TRhs: RandomVariable,
{
    type Output = IndependentJoint<Self, Rhs, Matrix, TRhs, WishartParams>;

    fn mul(self, rhs: Rhs) -> Self::Output {
        IndependentJoint::new(self, rhs)
    }
}

impl<Rhs, URhs> BitAnd<Rhs> for Wishart
where
    Rhs: Distribution<Value = WishartParams, Condition = URhs>,
    URhs: RandomVariable,
{
    type Output = DependentJoint<Self, Rhs, Matrix, WishartParams, URhs>;

    fn bitand(self, rhs: Rhs) -> Self::Output {
        DependentJoint::new(self, rhs)
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
