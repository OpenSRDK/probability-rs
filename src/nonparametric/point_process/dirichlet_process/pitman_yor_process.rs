use super::{
    BaseDirichletProcessParams, DirichletProcess, DirichletProcessParams, DirichletRandomMeasure,
};
use crate::DistributionError;
use crate::{DependentJoint, Distribution, IndependentJoint, RandomVariable};
use rand::prelude::*;
use std::marker::PhantomData;
use std::{ops::BitAnd, ops::Mul};

/// # Pitman-Yor process
#[derive(Clone, Debug)]
pub struct PitmanYorProcess<G0, T>
where
    G0: Distribution<T = T, U = ()>,
    T: RandomVariable,
{
    gibbs_iter: usize,
    max_len: usize,
    phantom: PhantomData<(G0, T)>,
}

impl<G0, T> PitmanYorProcess<G0, T>
where
    G0: Distribution<T = T, U = ()>,
    T: RandomVariable,
{
    pub fn new(gibbs_iter: usize, max_len: usize) -> Self {
        Self {
            gibbs_iter,
            max_len,
            phantom: PhantomData,
        }
    }
}

#[derive(thiserror::Error, Debug)]
pub enum PitmanYorProcessError {
    #[error("'d' must be greater than or equal to 0 and less than 1")]
    DMustBeGTE0AndLT1,
    #[error("Unknown error")]
    Unknown,
}

impl<G0, T> Distribution for PitmanYorProcess<G0, T>
where
    G0: Distribution<T = T, U = ()>,
    T: RandomVariable,
{
    type T = DirichletRandomMeasure<T>;
    type U = PitmanYorProcessParams<G0, T>;

    fn p(&self, x: &Self::T, theta: &Self::U) -> Result<f64, DistributionError> {
        let alpha = theta.alpha();
        let n = x.z.len() as f64;

        let p = x
            .w_theta()
            .iter()
            .map(|(wj, thetaj)| -> Result<_, DistributionError> {
                let init_p = (alpha + theta.d) / (n + alpha);
                let additional_p = (1..*wj)
                    .into_iter()
                    .map(|ni| (ni as f64 - theta.d) / (n + alpha))
                    .product::<f64>();
                let thetaj_p = theta.g0().distr.p(&thetaj, &())?;

                Ok(init_p * additional_p * thetaj_p)
            })
            .product::<Result<f64, _>>();

        p
    }

    fn sample(&self, theta: &Self::U, rng: &mut StdRng) -> Result<Self::T, DistributionError> {
        let n = self.max_len;
        let mut z = vec![0usize; n];
        let mut n_vec = vec![n; 1];
        let mut empty_cluster_minimum_index = n_vec.len();

        for _ in 0..self.gibbs_iter * n {
            let i = rng.gen_range(0..n);

            n_vec[z[i]] -= 1;
            if n_vec[z[i]] == 0 && z[i] < empty_cluster_minimum_index {
                empty_cluster_minimum_index = z[i];
            }

            let p = rng.gen_range(0.0..=1.0);
            let mut p_sum = 0.0;

            let mut result_j = empty_cluster_minimum_index;

            for (j, &nj) in n_vec.iter().enumerate().filter(|(_, &nj)| nj != 0) {
                p_sum += (nj as f64 - theta.d) / (n as f64 + theta.alpha());
                if p < p_sum {
                    result_j = j;
                    break;
                }
            }
            z[i] = result_j;

            if n_vec.len() == z[i] {
                n_vec.push(1);
            } else {
                n_vec[z[i]] += 1;
            }
        }

        let w_theta = n_vec
            .iter()
            .filter(|&nj| *nj != 0)
            .map(|&nj| -> Result<_, DistributionError> {
                let w = nj;
                let theta = theta.g0().distr.sample(&(), rng)?;
                Ok((w, theta))
            })
            .collect::<Result<Vec<_>, _>>()?;

        Self::z_compaction(&mut z, &n_vec);

        Ok(DirichletRandomMeasure::new(w_theta, z))
    }
}

impl<G0, T> DirichletProcess<PitmanYorProcessParams<G0, T>, G0, T> for PitmanYorProcess<G0, T>
where
    G0: Distribution<T = T, U = ()>,
    T: RandomVariable,
{
}

#[derive(Clone, Debug)]
pub struct PitmanYorProcessParams<G0, T>
where
    G0: Distribution<T = T, U = ()>,
    T: RandomVariable,
{
    base: BaseDirichletProcessParams<G0, T>,
    d: f64,
}

impl<G0, T> PitmanYorProcessParams<G0, T>
where
    G0: Distribution<T = T, U = ()>,
    T: RandomVariable,
{
    /// - `d`: 0 ≦ d < 1. If it is zero, Pitman-Yor process means Chinese restaurant process.
    pub fn new(base: BaseDirichletProcessParams<G0, T>, d: f64) -> Result<Self, DistributionError> {
        if d < 0.0 || 1.0 <= d {
            return Err(DistributionError::InvalidParameters(
                PitmanYorProcessError::DMustBeGTE0AndLT1.into(),
            ));
        }

        Ok(Self { base, d })
    }

    pub fn base(&self) -> &BaseDirichletProcessParams<G0, T> {
        &self.base
    }

    pub fn d(&self) -> f64 {
        self.d
    }
}

impl<G0, T> DirichletProcessParams<G0, T> for PitmanYorProcessParams<G0, T>
where
    G0: Distribution<T = T, U = ()>,
    T: RandomVariable,
{
    fn alpha(&self) -> f64 {
        self.base.alpha()
    }

    fn g0(&self) -> &crate::nonparametric::BaselineMeasure<G0, T> {
        self.base.g0()
    }
}

impl<G0, T, Rhs, TRhs> Mul<Rhs> for PitmanYorProcess<G0, T>
where
    G0: Distribution<T = T, U = ()>,
    T: RandomVariable,
    Rhs: Distribution<T = TRhs, U = PitmanYorProcessParams<G0, T>>,
    TRhs: RandomVariable,
{
    type Output =
        IndependentJoint<Self, Rhs, DirichletRandomMeasure<T>, TRhs, PitmanYorProcessParams<G0, T>>;

    fn mul(self, rhs: Rhs) -> Self::Output {
        IndependentJoint::new(self, rhs)
    }
}

impl<G0, T, Rhs, URhs> BitAnd<Rhs> for PitmanYorProcess<G0, T>
where
    G0: Distribution<T = T, U = ()>,
    T: RandomVariable,
    Rhs: Distribution<T = PitmanYorProcessParams<G0, T>, U = URhs>,
    URhs: RandomVariable,
{
    type Output =
        DependentJoint<Self, Rhs, DirichletRandomMeasure<T>, PitmanYorProcessParams<G0, T>, URhs>;

    fn bitand(self, rhs: Rhs) -> Self::Output {
        DependentJoint::new(self, rhs)
    }
}

#[cfg(test)]
mod tests {
    use crate::distribution::Distribution;
    use crate::nonparametric::*;
    use crate::*;
    use rand::prelude::*;
    #[test]
    fn it_works() {
        let mut rng = StdRng::from_seed([1; 32]);

        let mu1 = 1.0;
        let sigma1 = 103.0;
        let x1 = (0..100)
            .into_iter()
            .map(|_| {
                Normal
                    .sample(&NormalParams::new(mu1, sigma1).unwrap(), &mut rng)
                    .unwrap()
            })
            .collect::<Vec<_>>();

        let mu2 = 5.0;
        let sigma2 = 10.0;
        let x2 = (0..100)
            .into_iter()
            .map(|_| {
                Normal
                    .sample(&NormalParams::new(mu2, sigma2).unwrap(), &mut rng)
                    .unwrap()
            })
            .collect::<Vec<_>>();

        let x = [x1, x2].concat();
        let n = x.len();

        let alpha = 0.5;
        let d = 0.5;

        let distr = InstantDistribution::new(
            &|x: &Vec<f64>, theta: &DirichletRandomMeasure<NormalParams>| {
                x.iter()
                    .enumerate()
                    .map(|(i, xi)| Normal.p(xi, &theta.w_theta[theta.z[i]].1))
                    .product::<Result<f64, _>>()
            },
            &|theta, rng| {
                (0..theta.z.len())
                    .into_iter()
                    .map(|i| Normal.sample(&theta.w_theta[theta.z[i]].1, rng))
                    .collect::<Result<Vec<_>, _>>()
            },
        );

        let g0 = BaselineMeasure::new(InstantDistribution::new(
            &|_: &NormalParams, _: &()| Ok(0.1),
            &|_, rng| {
                NormalParams::new(
                    10.0 * rng.gen_range(0.0..=1.0),
                    10.0 * rng.gen_range(0.0..=1.0) + 10.0,
                )
            },
        ));
        let pyp = PitmanYorProcess::<InstantDistribution<_, ()>, NormalParams>::new(4, n);
        let pyp_params =
            PitmanYorProcessParams::new(BaseDirichletProcessParams::new(alpha, g0).unwrap(), d)
                .unwrap();
        let condition = |_: &()| Ok(pyp_params.clone());

        let pyp_conditioned = pyp.condition(&condition);

        let mh_sampler = MetropolisHastingsSampler::new(&x, &distr, &pyp_conditioned, ());

        let mut most_likely_result = (
            0.0,
            DirichletRandomMeasure::<NormalParams>::new(vec![], vec![]),
        );

        for _ in 0..20 {
            let g = mh_sampler
                .sample(5, pyp_conditioned.sample(&(), &mut rng).unwrap(), &mut rng)
                .unwrap();
            let p = distr.p(&x, &g).unwrap() * pyp_conditioned.p(&g, &()).unwrap();

            if most_likely_result.0 < p {
                most_likely_result = (p, g);
            }
        }

        println!("クラスタ数:{}", most_likely_result.1.z().len());
    }
}
