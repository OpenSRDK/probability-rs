use crate::nonparametric::PitmanYorDP;
use crate::DistributionError;
use crate::{DependentJoint, Distribution, IndependentJoint, RandomVariable};
use rand::prelude::*;
use std::{ops::BitAnd, ops::Mul};

use super::PitmanYorDPParams;

/// # Pitman-Yor dirichlet process
#[derive(Clone, Debug)]
pub struct PitmanYorDPGibbs<'a, T, U, D, G0>
where
    T: RandomVariable,
    U: RandomVariable,
    D: Distribution<T = T, U = U>,
    G0: Distribution<T = U, U = ()>,
{
    alpha: f64,
    d: f64,
    i: usize,
    x: &'a Vec<T>,
    theta: &'a Vec<U>,
    distr: D,
    g0: G0,
}

impl<'a, T, U, D, G0> PitmanYorDPGibbs<'a, T, U, D, G0>
where
    T: RandomVariable,
    U: RandomVariable,
    D: Distribution<T = T, U = U>,
    G0: Distribution<T = U, U = ()>,
{
    pub fn new(
        alpha: f64,
        d: f64,
        i: usize,
        x: &'a Vec<T>,
        theta: &'a Vec<U>,
        distr: D,
        g0: G0,
    ) -> Self {
        Self {
            alpha,
            d,
            i,
            x,
            theta,
            distr,
            g0,
        }
    }
}

impl<'a, T, U, D, G0> Distribution for PitmanYorDPGibbs<'a, T, U, D, G0>
where
    T: RandomVariable,
    U: RandomVariable,
    D: Distribution<T = T, U = U>,
    G0: Distribution<T = U, U = ()>,
{
    type T = usize;
    type U = Vec<usize>;

    fn p(&self, x: &Self::T, theta: &Self::U) -> Result<f64, DistributionError> {
        let mut params = PitmanYorDPParams::new(self.alpha, self.d, vec![])?;
        params.z_mut().copy_from_slice(theta);

        PitmanYorDP.p(x, &params)
    }

    fn sample(&self, theta: &Self::U, rng: &mut StdRng) -> Result<Self::T, DistributionError> {
        let mut params = PitmanYorDPParams::new(self.alpha, self.d, vec![])?;
        params.z_mut().copy_from_slice(theta);

        let n_vec = params.clusters();
        let max_k = n_vec.len();

        let p = rng.gen_range(0.0..1.0);
        let mut p_sum = 0.0;

        for k in 0..max_k {
            let pitman_yor_p = PitmanYorDP.p(&k, &params)?;
            p_sum += pitman_yor_p * self.distr.p(&self.x[self.i], &self.theta[k])?;
            if p < p_sum {
                return Ok(k);
            }
        }

        Ok(max_k)
    }
}

impl<'a, T, U, D, G0, Rhs, TRhs> Mul<Rhs> for PitmanYorDPGibbs<'a, T, U, D, G0>
where
    T: RandomVariable,
    U: RandomVariable,
    D: Distribution<T = T, U = U>,
    G0: Distribution<T = U, U = ()>,
    Rhs: Distribution<T = TRhs, U = Vec<usize>>,
    TRhs: RandomVariable,
{
    type Output = IndependentJoint<Self, Rhs, usize, TRhs, Vec<usize>>;

    fn mul(self, rhs: Rhs) -> Self::Output {
        IndependentJoint::new(self, rhs)
    }
}

impl<'a, T, U, D, G0, Rhs, URhs> BitAnd<Rhs> for PitmanYorDPGibbs<'a, T, U, D, G0>
where
    T: RandomVariable,
    U: RandomVariable,
    D: Distribution<T = T, U = U>,
    G0: Distribution<T = U, U = ()>,
    Rhs: Distribution<T = Vec<usize>, U = URhs>,
    URhs: RandomVariable,
{
    type Output = DependentJoint<Self, Rhs, usize, Vec<usize>, URhs>;

    fn bitand(self, rhs: Rhs) -> Self::Output {
        DependentJoint::new(self, rhs)
    }
}

#[cfg(test)]
mod tests {
    use crate::*;
    use rand::prelude::*;

    use super::PitmanYorDPGibbs;

    #[test]
    fn it_works() {
        let x = vec![];
        let n = x.len();

        let alpha = 0.5;
        let d = 0.5;

        let mut z = (0..n).into_iter().collect::<Vec<_>>();
        let theta = vec![NormalParams::new(0.0, 1.0).unwrap(); n];

        let sampler = GibbsSampler::new(
            (0..n)
                .into_iter()
                .map(|i| {
                    PitmanYorDPGibbs::new(
                        alpha,
                        d,
                        i,
                        &x,
                        &theta,
                        Normal,
                        InstantDistribution::new(&|x: &NormalParams, _| Ok(x.mu()), &|_, rng| {
                            let mu = rng.gen_range(0.0..=1.0);
                            NormalParams::new(mu, 10.0 * mu)
                        }),
                    )
                })
                .collect(),
        );

        let iter = 10;
        let mut rng = StdRng::from_seed([1; 32]);

        for _ in 0..iter {
            z = sampler.step_sample(z, &mut rng).unwrap();
        }
    }
}
