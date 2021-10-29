use super::PitmanYorProcessParams;
use crate::nonparametric::*;
use crate::DistributionError;
use crate::{DependentJoint, Distribution, IndependentJoint, RandomVariable};
use rand::prelude::*;
use std::{ops::BitAnd, ops::Mul};

/// # Pitman-Yor dirichlet process
#[derive(Clone, Debug)]
pub struct PitmanYorGibbs<G0, T>
where
    G0: Distribution<T = T, U = ()>,
    T: RandomVariable,
{
    params: PitmanYorProcessParams<G0, T>,
}

impl<G0, T> PitmanYorGibbs<G0, T>
where
    G0: Distribution<T = T, U = ()>,
    T: RandomVariable,
{
    pub fn new(params: PitmanYorProcessParams<G0, T>) -> Self {
        Self { params }
    }
}

impl<G0, T> Distribution for PitmanYorGibbs<G0, T>
where
    G0: Distribution<T = T, U = ()>,
    T: RandomVariable,
{
    type T = usize;
    type U = Vec<usize>;

    fn p(&self, x: &Self::T, theta: &Self::U) -> Result<f64, DistributionError> {
        let alpha = self.params.alpha();
        let d = self.params.d();
        let k = *x;
        let z = theta;
        let n = z.len();
        let n_vec = clusters(z);

        if n_vec[k] == 0 {
            return Ok(0.0);
        }

        if k < n_vec.len() {
            Ok((n_vec[k] as f64 - d) / (n as f64 + alpha))
        } else {
            Ok((alpha + d) / (n as f64 + alpha))
        }
    }

    fn sample(&self, theta: &Self::U, rng: &mut StdRng) -> Result<Self::T, DistributionError> {
        let n_vec = clusters(theta);
        let max_k = n_vec.len();

        let p = rng.gen_range(0.0..1.0);
        let mut p_sum = 0.0;

        for k in 0..max_k {
            p_sum += self.p(&k, theta)?;
            if p < p_sum {
                return Ok(k);
            }
        }

        Ok(max_k)
    }
}

impl<G0, T, Rhs, TRhs> Mul<Rhs> for PitmanYorGibbs<G0, T>
where
    G0: Distribution<T = T, U = ()>,
    T: RandomVariable,
    Rhs: Distribution<T = TRhs, U = Vec<usize>>,
    TRhs: RandomVariable,
{
    type Output = IndependentJoint<Self, Rhs, usize, TRhs, Vec<usize>>;

    fn mul(self, rhs: Rhs) -> Self::Output {
        IndependentJoint::new(self, rhs)
    }
}

impl<G0, T, Rhs, URhs> BitAnd<Rhs> for PitmanYorGibbs<G0, T>
where
    G0: Distribution<T = T, U = ()>,
    T: RandomVariable,
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

        let mut z = (0..n).into_iter().collect::<Vec<_>>();
        let pyp_params =
            PitmanYorProcessParams::new(BaseDirichletProcessParams::new(alpha, g0).unwrap(), d)
                .unwrap();
        let pyp = PitmanYorGibbs::<InstantDistribution<_, ()>, NormalParams>::new(pyp_params);
        let sample_rng = rng.gen_range(0..n);
        let sample_distr = GibbsSampler.step_sample(sample_rng, z, pyp, &mut rng);
        //ここからやっていこう
        let mh_sampler = MetropolisHastingsSampler::new(&x, &distr, &sample_distr, &());

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
