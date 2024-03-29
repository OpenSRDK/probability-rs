use crate::{
    ConditionDifferentiableDistribution, DistributionError, SamplableDistribution,
    ValueDifferentiableDistribution,
};
use crate::{DependentJoint, Distribution, IndependentJoint, RandomVariable};
use rand::prelude::*;
use rand_distr::Beta as RandBeta;
use special::Gamma;
use std::{ops::BitAnd, ops::Mul};

/// Beta distribution
#[derive(Clone, Debug)]
pub struct Beta;

#[derive(thiserror::Error, Debug)]
pub enum BetaError {
    #[error("'α' must be positive")]
    AlphaMustBePositive,
    #[error("'β' must be positive")]
    BetaMustBePositive,
}

impl Distribution for Beta {
    type Value = f64;
    type Condition = BetaParams;

    fn p_kernel(&self, x: &Self::Value, theta: &Self::Condition) -> Result<f64, DistributionError> {
        let alpha = theta.alpha();
        let beta = theta.beta();

        Ok(x.powf(alpha - 1.0) * (1.0 - x).powf(beta - 1.0))
    }
}

impl ValueDifferentiableDistribution for Beta {
    fn ln_diff_value(
        &self,
        x: &Self::Value,
        theta: &Self::Condition,
    ) -> Result<Vec<f64>, DistributionError> {
        let alpha = theta.alpha();
        let beta = theta.beta();
        let f_x = (alpha - 1.0) / x - (beta - 1.0) / (1.0 - x);
        Ok(vec![f_x])
    }
}

impl ConditionDifferentiableDistribution for Beta {
    fn ln_diff_condition(
        &self,
        x: &Self::Value,
        theta: &Self::Condition,
    ) -> Result<Vec<f64>, DistributionError> {
        let alpha = theta.alpha();
        let beta = theta.beta();
        let f_alpha = x.ln() - alpha.digamma() + (alpha + beta).digamma();
        let f_beta = (1.0 - x).ln() - beta.digamma() + (alpha + beta).digamma();
        Ok(vec![f_alpha, f_beta])
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct BetaParams {
    alpha: f64,
    beta: f64,
}

impl BetaParams {
    pub fn new(alpha: f64, beta: f64) -> Result<Self, DistributionError> {
        if alpha <= 0.0 {
            return Err(DistributionError::InvalidParameters(
                BetaError::AlphaMustBePositive.into(),
            ));
        }
        if alpha <= 0.0 {
            return Err(DistributionError::InvalidParameters(
                BetaError::BetaMustBePositive.into(),
            ));
        }

        Ok(Self { alpha, beta })
    }

    pub fn alpha(&self) -> f64 {
        self.alpha
    }

    pub fn beta(&self) -> f64 {
        self.beta
    }
}

impl RandomVariable for BetaParams {
    type RestoreInfo = ();

    fn transform_vec(&self) -> (Vec<f64>, Self::RestoreInfo) {
        (vec![self.alpha, self.beta], ())
    }

    fn len(&self) -> usize {
        2usize
    }

    fn restore(v: &[f64], _: &Self::RestoreInfo) -> Result<Self, DistributionError> {
        Self::new(v[0], v[1])
    }
}

impl<Rhs, TRhs> Mul<Rhs> for Beta
where
    Rhs: Distribution<Value = TRhs, Condition = BetaParams>,
    TRhs: RandomVariable,
{
    type Output = IndependentJoint<Self, Rhs, f64, TRhs, BetaParams>;

    fn mul(self, rhs: Rhs) -> Self::Output {
        IndependentJoint::new(self, rhs)
    }
}

impl<Rhs, URhs> BitAnd<Rhs> for Beta
where
    Rhs: Distribution<Value = BetaParams, Condition = URhs>,
    URhs: RandomVariable,
{
    type Output = DependentJoint<Self, Rhs, f64, BetaParams, URhs>;

    fn bitand(self, rhs: Rhs) -> Self::Output {
        DependentJoint::new(self, rhs)
    }
}

impl SamplableDistribution for Beta {
    fn sample(
        &self,
        theta: &Self::Condition,
        rng: &mut dyn RngCore,
    ) -> Result<Self::Value, DistributionError> {
        let alpha = theta.alpha();
        let beta = theta.beta();

        let beta = match RandBeta::new(alpha, beta) {
            Ok(v) => Ok(v),
            Err(e) => Err(DistributionError::Others(e.into())),
        }?;

        Ok(rng.sample(beta))
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        Beta, BetaParams, ConditionDifferentiableDistribution, Distribution, SamplableDistribution,
        ValueDifferentiableDistribution,
    };
    use rand::prelude::*;

    #[test]
    fn it_works() {
        let n = Beta;
        let mut rng = StdRng::from_seed([1; 32]);

        let alpha = 2.0;
        let beta = 3.0;

        let x = n
            .sample(&BetaParams::new(alpha, beta).unwrap(), &mut rng)
            .unwrap();

        println!("{}", x);
    }
    #[test]
    fn it_works2() {
        let n = Beta;

        let alpha = 2.0;
        let beta = 3.0;

        let x = 0.5;

        let f = n.ln_diff_value(&x, &BetaParams::new(alpha, beta).unwrap());
        println!("{:#?}", f);
    }

    #[test]
    fn it_works_3() {
        let n = Beta;

        let alpha = 2.0;
        let beta = 3.0;

        let x = 0.5;

        let f = n.ln_diff_condition(&x, &BetaParams::new(alpha, beta).unwrap());
        println!("{:#?}", f);
    }
}
