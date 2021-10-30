use crate::nonparametric::*;
use crate::DistributionError;
use crate::{DependentJoint, Distribution, IndependentJoint, RandomVariable};
use rand::prelude::*;
use std::marker::PhantomData;
use std::{ops::BitAnd, ops::Mul};

/// # Pitman-Yor process
#[derive(Clone, Debug)]
pub struct PitmanYorGibbs<'a> {
    phantom: &'a PhantomData<()>,
}

impl<'a> PitmanYorGibbs<'a> {
    pub fn new() -> Self {
        Self {
            phantom: &PhantomData,
        }
    }
}

impl<'a> Distribution for PitmanYorGibbs<'a> {
    type T = u32;
    type U = PitmanYorGibbsParams<'a>;

    fn p(&self, x: &Self::T, theta: &Self::U) -> Result<f64, DistributionError> {
        let alpha = theta.base.alpha;
        let d = theta.base.d;
        let k = *x;
        let s = theta.s;
        let n = s.len() - 1;
        let n_map = clusters(s);
        let mut nk = match n_map.get(&k) {
            Some(&v) => v,
            None => 0,
        };

        if s[theta.remove_index] == k {
            nk -= 1
        }

        if nk != 0 {
            return Ok((nk as f64 - d) / (n as f64 + alpha));
        }

        Ok((alpha + d) / (n as f64 + alpha))
    }

    /// 0 means new cluster. However, you can't use 0 for `s` so use another value which will not conflict.
    fn sample(&self, theta: &Self::U, rng: &mut StdRng) -> Result<Self::T, DistributionError> {
        let alpha = theta.base.alpha;
        let d = theta.base.d;
        let s = theta.s;
        let n = s.len();
        let n_map = clusters(s);

        let p = rng.gen_range(0.0..1.0);
        let mut p_sum = 0.0;

        for (&k, &nk) in n_map.iter() {
            if s[theta.remove_index] == k && nk == 1 {
                continue;
            }
            p_sum += (nk as f64 - d) / (n as f64 + alpha);
            if p < p_sum {
                return Ok(k);
            }
        }

        Ok(0)
    }
}

#[derive(Clone, Debug)]
pub struct PitmanYorGibbsParams<'a> {
    base: PitmanYorProcessParams,
    s: &'a [u32],
    remove_index: usize,
}

impl<'a> PitmanYorGibbsParams<'a> {
    /// - `d`: 0 ≦ d < 1. If it is zero, Pitman-Yor process means Chinese restaurant process.
    pub fn new(
        base: PitmanYorProcessParams,
        s: &'a [u32],
        remove_index: usize,
    ) -> Result<Self, DistributionError> {
        for &si in s.iter() {
            if si == 0 {
                return Err(DistributionError::InvalidParameters(
                    PitmanYorProcessError::SMustBePositive.into(),
                ));
            }
        }

        if s.len() <= remove_index {
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

impl<'a, Rhs, TRhs> Mul<Rhs> for PitmanYorGibbs<'a>
where
    Rhs: Distribution<T = TRhs, U = PitmanYorGibbsParams<'a>>,
    TRhs: RandomVariable,
{
    type Output = IndependentJoint<Self, Rhs, u32, TRhs, PitmanYorGibbsParams<'a>>;

    fn mul(self, rhs: Rhs) -> Self::Output {
        IndependentJoint::new(self, rhs)
    }
}

impl<'a, Rhs, URhs> BitAnd<Rhs> for PitmanYorGibbs<'a>
where
    Rhs: Distribution<T = PitmanYorGibbsParams<'a>, U = URhs>,
    URhs: RandomVariable,
{
    type Output = DependentJoint<Self, Rhs, u32, PitmanYorGibbsParams<'a>, URhs>;

    fn bitand(self, rhs: Rhs) -> Self::Output {
        DependentJoint::new(self, rhs)
    }
}
