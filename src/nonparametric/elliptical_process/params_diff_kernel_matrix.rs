use crate::opensrdk_linear_algebra::*;
use opensrdk_kernel_method::*;
use rayon::prelude::*;

pub fn params_diff_kernel_matrix<T>(
    kernel: &impl ParamsDifferentiableKernel<T>,
    params: &[f64],
    x: &[T],
    x_prime: &[T],
) -> Result<Vec<Matrix>, KernelError>
where
    T: Value,
{
    let m = x.len();
    let n = x_prime.len();

    let n_params = params.len();

    let params_diff_mat = &mut vec![Matrix::new(m, n); n_params];

    for i in 0..n {
        for j in 0..m {
            let diff_params = kernel.ln_diff_params(params, &x[i], &x_prime[j]).unwrap();
            for k in 0..n_params {
                params_diff_mat[k][(i, j)] = diff_params[k];
            }
        }
    }

    let result = params_diff_mat.clone();

    Ok(result)
}

#[cfg(test)]
mod tests {
    use crate::{
        nonparametric::{params_diff_kernel_matrix, BaseEllipticalProcessParams},
        ConditionDifferentiableDistribution, Distribution, ExactMultivariateNormalParams,
        MultivariateNormal, ValueDifferentiableDistribution,
    };
    use opensrdk_kernel_method::*;
    use opensrdk_linear_algebra::{pp::trf::PPTRF, *};
    use rand::prelude::*;
    use rand_distr::StandardNormal;
    use rayon::prelude::{IntoParallelRefIterator, ParallelIterator};

    #[test]
    fn it_works() {
        let mut _rng = StdRng::from_seed([1; 32]);

        let samples = samples(7);
        let x = samples.par_iter().map(|v| vec![v.0]).collect::<Vec<_>>();
        let y = samples.par_iter().map(|v| v.1).collect::<Vec<_>>();
        let y2 = vec![1.0; y.len()];
        let kernel = RBF;
        let kernel_len = kernel.params_len();
        let theta = vec![0.8; kernel_len];
        let sigma = 2.0;
        let base = &BaseEllipticalProcessParams::new(kernel, x, theta, sigma)
            .unwrap()
            .exact(&y)
            .unwrap();
        let kxx = params_diff_kernel_matrix(
            &base.base.kernel,
            &vec![1.8; kernel_len],
            &base.base.x,
            &base.base.x,
        )
        .unwrap();

        println!("{:#?}", kxx);
    }

    fn func(x: f64) -> f64 {
        0.1 * x + x.sin() + 2.0 * (-x.powi(2)).exp()
    }

    fn samples(size: usize) -> Vec<(f64, f64)> {
        let mut rng = StdRng::from_seed([1; 32]);
        let mut rng2 = StdRng::from_seed([32; 32]);

        (0..size)
            .into_iter()
            .map(|_| {
                let x = rng2.gen_range(-8.0..=8.0);
                let y = func(x) + rng.sample::<f64, _>(StandardNormal);

                (x, y)
            })
            .collect()
    }
}
