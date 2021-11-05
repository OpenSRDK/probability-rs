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
    G0: Distribution<T = TH, U = ()>,
    TH: RandomVariable,
{
    phantom: &'a PhantomData<(G0, TH)>,
}

impl<'a, G0, TH> PitmanYorGibbs<'a, G0, TH>
where
    G0: Distribution<T = TH, U = ()>,
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
    G0: Distribution<T = TH, U = ()>,
    TH: RandomVariable,
{
    type T = PitmanYorGibbsSample;
    type U = PitmanYorGibbsParams<'a, G0, TH>;

    fn fk(&self, x: &Self::T, theta: &Self::U) -> Result<f64, DistributionError> {
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

    fn sample(
        &self,
        _theta: &Self::U,
        _rng: &mut dyn RngCore,
    ) -> Result<Self::T, DistributionError> {
        todo!()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PitmanYorGibbsSample {
    Existing(u32),
    New,
}

#[derive(Clone, Debug)]
pub struct PitmanYorGibbsParams<'a, G0, TH>
where
    G0: Distribution<T = TH, U = ()>,
    TH: RandomVariable,
{
    base: &'a PitmanYorProcessParams<G0, TH>,
    s_inv: &'a HashMap<u32, HashSet<usize>>,
    n: usize,
}

impl<'a, G0, TH> PitmanYorGibbsParams<'a, G0, TH>
where
    G0: Distribution<T = TH, U = ()>,
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

impl<'a, G0, TH, Rhs, TRhs> Mul<Rhs> for PitmanYorGibbs<'a, G0, TH>
where
    G0: Distribution<T = TH, U = ()>,
    TH: RandomVariable,
    Rhs: Distribution<T = TRhs, U = PitmanYorGibbsParams<'a, G0, TH>>,
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
    G0: Distribution<T = TH, U = ()>,
    TH: RandomVariable,
    Rhs: Distribution<T = PitmanYorGibbsParams<'a, G0, TH>, U = URhs>,
    URhs: RandomVariable,
{
    type Output =
        DependentJoint<Self, Rhs, PitmanYorGibbsSample, PitmanYorGibbsParams<'a, G0, TH>, URhs>;

    fn bitand(self, rhs: Rhs) -> Self::Output {
        DependentJoint::new(self, rhs)
    }
}
