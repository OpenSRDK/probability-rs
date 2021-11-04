use crate::nonparametric::*;
use crate::DistributionError;
use crate::{DependentJoint, Distribution, IndependentJoint, RandomVariable};
use rand::prelude::*;
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
        let s = theta.s.s();
        let n = s.len().max(1) - 1;
        let clusters_len = theta.s.clusters_len();

        match *x {
            PitmanYorGibbsSample::Existing(k) => {
                let mut nk = theta.s.n(k);

                if s[theta.remove_index] == k {
                    nk -= 1
                }

                if nk != 0 {
                    return Ok((nk as f64 - d) / (n as f64 + alpha));
                }

                Ok((alpha + clusters_len as f64 * d) / (n as f64 + alpha))
            }
            PitmanYorGibbsSample::New => Ok((alpha + clusters_len as f64 * d) / (n as f64 + alpha)),
        }
    }

    fn sample(&self, theta: &Self::U, rng: &mut dyn RngCore) -> Result<Self::T, DistributionError> {
        let alpha = theta.base.alpha;
        let d = theta.base.d;
        let s = theta.s.s();
        let n = s.len().max(1) - 1;
        let s_inv = theta.s.s_inv();

        let p = rng.gen_range(0.0..=1.0);
        let mut p_sum = 0.0;

        for (&k, indice) in s_inv.iter() {
            let mut nk = indice.len();
            if s[theta.remove_index] == k {
                nk -= 1;
            }
            if nk == 0 {
                continue;
            }
            p_sum += (nk as f64 - d) / (n as f64 + alpha);
            if p < p_sum {
                return Ok(PitmanYorGibbsSample::Existing(k));
            }
        }

        Ok(PitmanYorGibbsSample::New)
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
    s: &'a ClusterSwitch<TH>,
    remove_index: usize,
}

impl<'a, G0, TH> PitmanYorGibbsParams<'a, G0, TH>
where
    G0: Distribution<T = TH, U = ()>,
    TH: RandomVariable,
{
    /// - `d`: 0 ≦ d < 1. If it is zero, Pitman-Yor process means Chinese restaurant process.
    pub fn new(
        base: &'a PitmanYorProcessParams<G0, TH>,
        s: &'a ClusterSwitch<TH>,
        remove_index: usize,
    ) -> Result<Self, DistributionError> {
        if s.s().len() <= remove_index {
            return Err(DistributionError::InvalidParameters(
                PitmanYorProcessError::RemoveIndexOutOfRange.into(),
            ));
        }

        Ok(Self {
            base,
            s,
            remove_index,
        })
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
