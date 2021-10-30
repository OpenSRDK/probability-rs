use super::DirichletProcessError;
use crate::{Beta, BetaParams, DistributionError};
use crate::{DependentJoint, Distribution, IndependentJoint, RandomVariable};
use rand::prelude::*;
use std::{ops::BitAnd, ops::Mul};

/// # Stick breaking process
/// https://papers.ssrn.com/sol3/papers.cfm?abstract_id=945330
#[derive(Clone, Debug)]
pub struct StickBreakingProcess;

impl Distribution for StickBreakingProcess {
    type T = Vec<f64>;
    type U = StickBreakingProcessParams;

    fn p(&self, x: &Self::T, theta: &Self::U) -> Result<f64, DistributionError> {
        let mut accumulated_w = 0.0;
        let mut accumulated_p = 1.0;
        let beta_params = BetaParams::new(1.0, theta.alpha)?;

        for &wi in x.iter() {
            let vi = wi / (1.0 - accumulated_w);

            accumulated_p *= Beta.p(&vi, &beta_params)?;
            accumulated_w += wi;
        }

        Ok(accumulated_p)
    }

    fn sample(&self, theta: &Self::U, rng: &mut StdRng) -> Result<Self::T, DistributionError> {
        todo!("{:?}{:?}", theta, rng);
    }
}

#[derive(Clone, Debug)]
pub struct StickBreakingProcessParams {
    alpha: f64,
}

impl StickBreakingProcessParams {
    pub fn new(alpha: f64) -> Result<Self, DistributionError> {
        if alpha <= 0.0 {
            return Err(DistributionError::InvalidParameters(
                DirichletProcessError::AlphaMustBePositive.into(),
            ));
        }
        Ok(Self { alpha })
    }
}

impl<Rhs, TRhs> Mul<Rhs> for StickBreakingProcess
where
    Rhs: Distribution<T = TRhs, U = StickBreakingProcessParams>,
    TRhs: RandomVariable,
{
    type Output = IndependentJoint<Self, Rhs, Vec<f64>, TRhs, StickBreakingProcessParams>;

    fn mul(self, rhs: Rhs) -> Self::Output {
        IndependentJoint::new(self, rhs)
    }
}

impl<Rhs, URhs> BitAnd<Rhs> for StickBreakingProcess
where
    Rhs: Distribution<T = StickBreakingProcessParams, U = URhs>,
    URhs: RandomVariable,
{
    type Output = DependentJoint<Self, Rhs, Vec<f64>, StickBreakingProcessParams, URhs>;

    fn bitand(self, rhs: Rhs) -> Self::Output {
        DependentJoint::new(self, rhs)
    }
}
