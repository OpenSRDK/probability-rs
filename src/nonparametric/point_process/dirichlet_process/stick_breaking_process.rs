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
    type Value = Vec<f64>;
    type Condition = StickBreakingProcessParams;

    fn fk(&self, x: &Self::Value, theta: &Self::Condition) -> Result<f64, DistributionError> {
        let mut accumulated_w = 0.0;
        let mut accumulated_p = 1.0;
        let beta_params = BetaParams::new(1.0, theta.alpha)?;

        for &wi in x.iter() {
            let vi = wi / (1.0 - accumulated_w);

            accumulated_p *= Beta.fk(&vi, &beta_params)?;
            accumulated_w += wi;
        }

        Ok(accumulated_p)
    }

    // fn sample(
    //     &self,
    //     theta: &Self::Condition,
    //     rng: &mut dyn RngCore,
    // ) -> Result<Self::Value, DistributionError> {
    //     rng.gen_range(0..1);
    //     todo!("{:?}", theta);
    // }
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

impl RandomVariable for StickBreakingProcessParams {
    type RestoreInfo = ();

    fn transform_vec(&self) -> (Vec<f64>, Self::RestoreInfo) {
        todo!()
    }

    fn len(&self) -> usize {
        todo!()
    }

    fn restore(v: &[f64], info: &Self::RestoreInfo) -> Result<Self, DistributionError> {
        todo!()
    }
}

impl<Rhs, TRhs> Mul<Rhs> for StickBreakingProcess
where
    Rhs: Distribution<Value = TRhs, Condition = StickBreakingProcessParams>,
    TRhs: RandomVariable,
{
    type Output = IndependentJoint<Self, Rhs, Vec<f64>, TRhs, StickBreakingProcessParams>;

    fn mul(self, rhs: Rhs) -> Self::Output {
        IndependentJoint::new(self, rhs)
    }
}

impl<Rhs, URhs> BitAnd<Rhs> for StickBreakingProcess
where
    Rhs: Distribution<Value = StickBreakingProcessParams, Condition = URhs>,
    URhs: RandomVariable,
{
    type Output = DependentJoint<Self, Rhs, Vec<f64>, StickBreakingProcessParams, URhs>;

    fn bitand(self, rhs: Rhs) -> Self::Output {
        DependentJoint::new(self, rhs)
    }
}
