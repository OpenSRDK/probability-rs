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
    use crate::distribution::Distribution;
    use crate::{nonparametric::*, *};
    use pitman_yor_dp::PitmanYorDP;
    use rand::prelude::*;
    use rand::seq::index::sample;

    use super::PitmanYorDPGibbs;

    #[test]
    fn it_works() {
        let n = Normal;
        let mut rng = StdRng::from_seed([1; 32]);

        let mu1 = 1.0;
        let sigma1 = 3.0;
        let x1 = (0..100)
            .into_iter()
            .map(|i| sample(Normal, &NormalParams::new(mu1, sigma1).unwrap(), &mut rng).unwrap())
            .collect::<Vec<_>>();

        let mu2 = 1.0;
        let sigma2 = 3.0;
        let x2 = (0..100)
            .into_iter()
            .map(|i| sample(Normal, &NormalParams::new(mu1, sigma1).unwrap(), &mut rng).unwrap())
            .collect::<Vec<_>>();

        let x = vec![];
        let n = x.len();

        let alpha = 0.5;
        let d = 0.5;

        let theta = vec![NormalParams::new(0.0, 1.0).unwrap(); n];
        let distr = Normal;
        let g0 = InstantDistribution::new(&|x: &NormalParams, _| Ok(x.mu()), &|_, rng| {
            let mu = rng.gen_range(0.0..=1.0);
            NormalParams::new(mu, 10.0 * mu)
        });
        let sampler = GibbsSampler::new(
            (0..n)
                .into_iter()
                .map(|i| PitmanYorDPGibbs::new(alpha, d, i, &x, &theta, distr.clone(), g0.clone()))
                .collect(),
        );

        let iter = 10;
        let mut rng = StdRng::from_seed([1; 32]);
        let mut p0 = 0.0;
        let mut i0 = 0usize;
        let mut z0 = vec![];
        let mut theta0 = vec![];

        for pattern in 0..3 {
            let mut z = (0..n).into_iter().collect::<Vec<_>>();
            for _ in 0..iter {
                z = sampler.step_sample(z, &mut rng).unwrap();
                //クラスタごとに、パラメータθを微調整する処理をここに書く
            }
            //3パターンのうち最も尤度が高いzを選ぶ処理をここに書く
            let params = PitmanYorDPParams::new(alpha, d, z.clone()).unwrap();
            let c = params.clusters_len();

            let p = (0..n)
                .into_iter()
                .map(|i| {
                    PitmanYorDP
                        .p(
                            &z[i],
                            &PitmanYorDPParams::new(alpha, d, z[0..i].to_vec()).unwrap(),
                        )
                        .unwrap()
                })
                .product::<f64>()
                * (0..c)
                    .into_iter()
                    .map(|j| g0.p(&theta[j], &()).unwrap())
                    .product::<f64>()
                * (0..n)
                    .into_iter()
                    .map(|i| distr.p(&x[i], &theta[z[i]]).unwrap())
                    .product::<f64>();
            if p0 < p {
                p0 = p;
                i0 = pattern;
                z0 = z;
                theta0 = theta.clone();
            }
        }
        println!("一番良いのは{}番目です！", i0);
    }
}
