use crate::nonparametric::*;
use crate::DistributionError;
use crate::{DependentJoint, Distribution, IndependentJoint, RandomVariable};
use rand::prelude::*;
use std::collections::HashMap;
use std::collections::HashSet;
use std::marker::PhantomData;
use std::{ops::BitAnd, ops::Mul};

/// # Pitman-Yor process
#[derive(Clone, Debug)]
pub struct PitmanYorGibbs<'a, G0, TH>
where
    G0: Distribution<Value = TH, Condition = ()>,
    TH: RandomVariable,
{
    phantom: &'a PhantomData<(G0, TH)>,
}

impl<'a, G0, TH> PitmanYorGibbs<'a, G0, TH>
where
    G0: Distribution<Value = TH, Condition = ()>,
    TH: RandomVariable,
{
    pub fn new() -> Self {
        Self {
            phantom: &PhantomData,
        }
    }
}

impl<'a, G0, TH> Distribution for PitmanYorGibbs<'a, G0, TH>
where
    G0: Distribution<Value = TH, Condition = ()>,
    TH: RandomVariable,
{
    type Value = PitmanYorGibbsSample;
    type Condition = PitmanYorGibbsParams<'a, G0, TH>;

    fn p_kernel(&self, x: &Self::Value, theta: &Self::Condition) -> Result<f64, DistributionError> {
        let alpha = theta.base.alpha;
        let d = theta.base.d;
        let n = theta.n;
        let clusters_len = theta.s_inv.len();

        match *x {
            PitmanYorGibbsSample::Existing(k) => {
                let nk = theta.s_inv.get(&k).unwrap_or(&HashSet::new()).len();

                if nk != 0 {
                    return Ok((nk as f64 - d) / (n as f64 + alpha));
                }

                Ok((alpha + clusters_len as f64 * d) / (n as f64 + alpha))
            }
            PitmanYorGibbsSample::New => Ok((alpha + clusters_len as f64 * d) / (n as f64 + alpha)),
        }
    }

    // fn sample(
    //     &self,
    //     _theta: &Self::Condition,
    //     _rng: &mut dyn RngCore,
    // ) -> Result<Self::Value, DistributionError> {
    //     todo!()
    // }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PitmanYorGibbsSample {
    Existing(u32),
    New,
}

impl RandomVariable for PitmanYorGibbsSample {
    type RestoreInfo = bool;

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

#[derive(Clone, Debug)]
pub struct PitmanYorGibbsParams<'a, G0, TH>
where
    G0: Distribution<Value = TH, Condition = ()>,
    TH: RandomVariable,
{
    base: &'a PitmanYorProcessParams<G0, TH>,
    s_inv: &'a HashMap<u32, HashSet<usize>>,
    n: usize,
}

impl<'a, G0, TH> PitmanYorGibbsParams<'a, G0, TH>
where
    G0: Distribution<Value = TH, Condition = ()>,
    TH: RandomVariable,
{
    /// - `d`: 0 ≦ d < 1. If it is zero, Pitman-Yor process means Chinese restaurant process.
    pub fn new(
        base: &'a PitmanYorProcessParams<G0, TH>,
        s_inv: &'a HashMap<u32, HashSet<usize>>,
        n: usize,
    ) -> Self {
        Self { base, s_inv, n }
    }
}

impl<'a, G0, TH> RandomVariable for PitmanYorGibbsParams<'a, G0, TH>
where
    G0: Distribution<Value = TH, Condition = ()>,
    TH: RandomVariable,
{
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

impl<'a, G0, TH, Rhs, TRhs> Mul<Rhs> for PitmanYorGibbs<'a, G0, TH>
where
    G0: Distribution<Value = TH, Condition = ()>,
    TH: RandomVariable,
    Rhs: Distribution<Value = TRhs, Condition = PitmanYorGibbsParams<'a, G0, TH>>,
    TRhs: RandomVariable,
{
    type Output =
        IndependentJoint<Self, Rhs, PitmanYorGibbsSample, TRhs, PitmanYorGibbsParams<'a, G0, TH>>;

    fn mul(self, rhs: Rhs) -> Self::Output {
        IndependentJoint::new(self, rhs)
    }
}

impl<'a, G0, TH, Rhs, URhs> BitAnd<Rhs> for PitmanYorGibbs<'a, G0, TH>
where
    G0: Distribution<Value = TH, Condition = ()>,
    TH: RandomVariable,
    Rhs: Distribution<Value = PitmanYorGibbsParams<'a, G0, TH>, Condition = URhs>,
    URhs: RandomVariable,
{
    type Output =
        DependentJoint<Self, Rhs, PitmanYorGibbsSample, PitmanYorGibbsParams<'a, G0, TH>, URhs>;

    fn bitand(self, rhs: Rhs) -> Self::Output {
        DependentJoint::new(self, rhs)
    }
}
