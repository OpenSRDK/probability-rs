use crate::Distribution;
use crate::DistributionError;
use crate::RandomVariable;
use opensrdk_linear_algebra::*;
use opensrdk_optimization::*;
use rand::prelude::*;
use rayon::prelude::*;
use std::error::Error;

pub trait VariationalInferenceDistribution<T>: Distribution<T = T, U = Vec<f64>>
where
    T: RandomVariable,
{
    /// Infer p(x|y) by approximating distribution q(x|θ) with p(y|x) and p(x).
    /// - `self`: q(x|θ)
    /// - `theta_len`: Length of the array of θ
    /// - `dqdtheta`: Differential of q(x|θ) with θ
    /// - `y`: y
    /// - `likelihood`: p(y|x)
    /// - `prior`: p(x)
    /// - `sample_batch`: Batch size of samples
    /// - `sample_total`: Total size of samples
    /// - `max_iter`: Max limitation of iteration count
    /// - `return`: θ
    fn variational_inference<U, DL, DP>(
        &self,
        theta_len: usize,
        dqdtheta: impl Fn(&T, &[f64]) -> Vec<f64> + Send + Sync,
        y: &U,
        likelihood: DL,
        prior: DP,
        sample_batch: usize,
        sample_total: usize,
        max_iter: usize,
    ) -> Result<Vec<f64>, Box<dyn Error>>
    where
        U: RandomVariable,
        DL: Distribution<T = U, U = T>,
        DP: Distribution<T = T, U = ()>,
    {
        let mut params = vec![0.0; theta_len];
        let mut rng = SeedableRng::from_rng(thread_rng())?;
        let x = (0..sample_total)
            .into_iter()
            .map(|_| prior.sample(&(), &mut rng))
            .collect::<Result<Vec<_>, _>>()?;

        SgdAdam::default().with_max_iter(max_iter).minimize(
            &mut params,
            &|indice, theta| {
                let theta = theta.to_vec();
                let x = indice.iter().map(|&i| &x[i]).collect::<Vec<_>>();

                let grad = x
                    .par_iter()
                    .map(|xi| -> Result<_, DistributionError> {
                        let dfdq = self.ln_p(xi, &theta)?
                            - likelihood.ln_p(y, xi)?
                            - prior.ln_p(xi, &())?;
                        Ok(dfdq * dqdtheta(xi, &theta).col_mat())
                    })
                    .try_reduce(
                        || vec![0.0; theta_len].col_mat(),
                        |acc, value| Ok(acc + value),
                    )
                    .unwrap_or(vec![f64::NAN; theta_len].col_mat())
                    .vec();

                grad.to_vec()
            },
            sample_batch,
            x.len(),
        );

        Ok(params)
    }
}

impl<D, T> VariationalInferenceDistribution<T> for D
where
    D: Distribution<T = T, U = Vec<f64>>,
    T: RandomVariable,
{
}
