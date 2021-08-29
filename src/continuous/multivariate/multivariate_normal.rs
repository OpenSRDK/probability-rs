use crate::{
    DependentJoint, Distribution, ExactEllipticalParams, IndependentJoint, RandomVariable,
};
use crate::{DistributionError, EllipticalParams};
use rand::prelude::*;
use rand_distr::StandardNormal;
use rayon::prelude::*;
use std::{f64::consts::PI, ops::BitAnd, ops::Mul};

/// # MultivariateNormal
/// ![tex](https://latex.codecogs.com/svg.latex?\mathcal%7BN%7D%28\mu%2C%20\Sigma%29)
#[derive(Clone, Debug)]
pub struct MultivariateNormal<T = ExactEllipticalParams>
where
    T: EllipticalParams;

#[derive(thiserror::Error, Debug)]
pub enum MultivariateNormalError {}

impl<T> Distribution for MultivariateNormal<T>
where
    T: EllipticalParams,
{
    type T = Vec<f64>;
    type U = T;

    fn p(&self, x: &Self::T, theta: &Self::U) -> Result<f64, DistributionError> {
        let x_mu = theta.x_mu(x)?;
        let n = x_mu.len() as f64;

        Ok(1.0 / ((2.0 * PI).powf(n / 2.0) * theta.lsigma_det())
            * (-1.0 / 2.0 * theta.x_mu_t_sigma_inv_x_mu(x_mu)?).exp())
    }

    fn sample(&self, theta: &Self::U, rng: &mut StdRng) -> Result<Self::T, DistributionError> {
        let z = (0..theta.z_len_for_sample())
            .into_iter()
            .map(|_| rng.sample(StandardNormal))
            .collect::<Vec<f64>>();

        let y = theta.sample_from_z(&z)?;

        Ok(y.vec())
    }
}

pub type ExactMultivariateNormalParams = ExactEllipticalParams;

impl<T, Rhs, TRhs> Mul<Rhs> for MultivariateNormal<T>
where
    T: EllipticalParams,
    Rhs: Distribution<T = TRhs, U = T>,
    TRhs: RandomVariable,
{
    type Output = IndependentJoint<Self, Rhs, Vec<f64>, TRhs, T>;

    fn mul(self, rhs: Rhs) -> Self::Output {
        IndependentJoint::new(self, rhs)
    }
}

impl<T, Rhs, URhs> BitAnd<Rhs> for MultivariateNormal<T>
where
    T: EllipticalParams,
    Rhs: Distribution<T = T, U = URhs>,
    URhs: RandomVariable,
{
    type Output = DependentJoint<Self, Rhs, Vec<f64>, T, URhs>;

    fn bitand(self, rhs: Rhs) -> Self::Output {
        DependentJoint::new(self, rhs)
    }
}

#[cfg(test)]
mod tests {
    use crate::{Distribution, ExactMultivariateNormalParams, MultivariateNormal};
    use opensrdk_linear_algebra::*;
    use rand::prelude::*;
    #[test]
    fn it_works() {
        let n = MultivariateNormal;
        let mut rng = StdRng::from_seed([1; 32]);

        let p = 6usize;
        let mu = vec![p as f64, 1.0, 2.0, 3.0, 4.0, 5.0];
        let lsigma = mat!(
           1.0,  0.0,  0.0,  0.0,  0.0,  0.0;
           2.0,  3.0,  0.0,  0.0,  0.0,  0.0;
           4.0,  5.0,  6.0,  0.0,  0.0,  0.0;
           7.0,  8.0,  9.0, 10.0,  0.0,  0.0;
          11.0, 12.0, 13.0, 14.0, 15.0,  0.0;
          16.0, 17.0, 18.0, 19.0, 20.0, 21.0
        );
        println!("{:#?}", lsigma);

        let x = n
            .sample(
                &ExactMultivariateNormalParams::new(mu, lsigma).unwrap(),
                &mut rng,
            )
            .unwrap();

        println!("{:#?}", x);
    }
}
