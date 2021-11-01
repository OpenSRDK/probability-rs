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
        let s = theta.s.s();
        let n = s.len().max(1) - 1;
        let clusters_len = theta.s.clusters_len();
        let mut nk = theta.s.n(k);

        if s[theta.remove_index] == k {
            nk -= 1
        }

        if nk != 0 {
            return Ok((nk as f64 - d) / (n as f64 + alpha));
        }

        Ok((alpha + clusters_len as f64 * d) / (n as f64 + alpha))
    }

    /// 0 means new cluster. However, you can't use 0 for `s` so use another value which will not conflict.
    fn sample(&self, theta: &Self::U, rng: &mut dyn RngCore) -> Result<Self::T, DistributionError> {
        let alpha = theta.base.alpha;
        let d = theta.base.d;
        let s = theta.s.s();
        let n = s.len().max(1) - 1;
        let s_inv = theta.s.s_inv();

        let p = rng.gen_range(0.0..1.0);
        let mut p_sum = 0.0;

        // todo: カテゴリ分布に変える？
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
                return Ok(k);
            }
        }

        let ret = if s_inv[&s[theta.remove_index]].len() == 1 {
            s[theta.remove_index]
        } else {
            0
        };

        Ok(ret)
    }
}

#[derive(Clone, Debug)]
pub struct PitmanYorGibbsParams<'a> {
    base: &'a PitmanYorProcessParams,
    s: &'a ClusterSwitch,
    remove_index: usize,
}

impl<'a> PitmanYorGibbsParams<'a> {
    /// - `d`: 0 ≦ d < 1. If it is zero, Pitman-Yor process means Chinese restaurant process.
    pub fn new(
        base: &'a PitmanYorProcessParams,
        s: &'a ClusterSwitch,
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
