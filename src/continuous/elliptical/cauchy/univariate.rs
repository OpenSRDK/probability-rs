use crate::{CauchyParams, DependentJoint, Distribution, IndependentJoint, RandomVariable};
use crate::{DistributionError, StudentT, StudentTParams};
use rand::prelude::*;
use std::{ops::BitAnd, ops::Mul};

/// Cauchy distribution
#[derive(Clone, Debug)]
pub struct Cauchy;

impl Distribution for Cauchy {
    type Value = f64;
    type Condition = CauchyParams;

    fn fk(&self, x: &Self::Value, theta: &Self::Condition) -> Result<f64, DistributionError> {
        let studentt_params = StudentTParams::new(1.0, theta.mu(), theta.sigma())?;

        StudentT.fk(x, &studentt_params)
    }

    fn sample(
        &self,
        theta: &Self::Condition,
        rng: &mut dyn RngCore,
    ) -> Result<Self::Value, DistributionError> {
        let studentt_params = StudentTParams::new(1.0, theta.mu(), theta.sigma())?;

        StudentT.sample(&studentt_params, rng)
    }
}

impl<Rhs, TRhs> Mul<Rhs> for Cauchy
where
    Rhs: Distribution<Value = TRhs, Condition = CauchyParams>,
    TRhs: RandomVariable,
{
    type Output = IndependentJoint<Self, Rhs, f64, TRhs, CauchyParams>;

    fn mul(self, rhs: Rhs) -> Self::Output {
        IndependentJoint::new(self, rhs)
    }
}

impl<Rhs, URhs> BitAnd<Rhs> for Cauchy
where
    Rhs: Distribution<Value = CauchyParams, Condition = URhs>,
    URhs: RandomVariable,
{
    type Output = DependentJoint<Self, Rhs, f64, CauchyParams, URhs>;

    fn bitand(self, rhs: Rhs) -> Self::Output {
        DependentJoint::new(self, rhs)
    }
}

#[cfg(test)]
mod tests {
    use crate::distribution::Distribution;
    use crate::*;
    use rand::prelude::*;

    #[test]
    fn it_works() {
        let n = Cauchy;
        let mut rng = StdRng::from_seed([1; 32]);

        let mu = 2.0;
        let sigma = 3.0;

        let x = n
            .sample(&CauchyParams::new(mu, sigma).unwrap(), &mut rng)
            .unwrap();

        println!("{}", x);
    }
}
