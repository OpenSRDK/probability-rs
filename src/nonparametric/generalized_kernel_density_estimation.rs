use crate::{
    nonparametric::kernel_matrix, Distribution, DistributionError, RandomVariable,
    SampleableDistribution,
};
use opensrdk_kernel_method::PositiveDefiniteKernel;
use rand::Rng;

#[derive(Clone, Debug)]
pub struct GeneralizedKernelDensity<S, A, K>
where
    S: RandomVariable,
    A: RandomVariable,
    K: PositiveDefiniteKernel<Vec<f64>>,
{
    history: Vec<(S, A)>,
    kernel: K,
    kernel_params: Vec<f64>, //基本的にはカーネル密度推定をしたいが、標本の空間が実数スカラー(wikiみたいなナイーブな例)ではなく任意の集合としたい
                             //sとa_othersの関係を学習したい
}

impl<S, A, K> GeneralizedKernelDensity<S, A, K>
where
    S: RandomVariable,
    A: RandomVariable,
    K: PositiveDefiniteKernel<Vec<f64>>,
{
    pub fn new(history: Vec<(S, A)>, kernel: K, kernel_params: Vec<f64>) -> Self {
        Self {
            history,
            kernel,
            kernel_params,
        }
    }
}

impl<S, A, K> Distribution for GeneralizedKernelDensity<S, A, K>
where
    S: RandomVariable,
    A: RandomVariable,
    K: PositiveDefiniteKernel<Vec<f64>>,
{
    type Value = A;
    type Condition = S;

    fn p_kernel(&self, x: &Self::Value, theta: &Self::Condition) -> Result<f64, DistributionError> {
        let v = std::iter::once([theta.transform_vec().0, x.transform_vec().0].concat())
            .chain(
                self.history
                    .iter()
                    .map(|e| [e.0.transform_vec().0, e.1.transform_vec().0].concat()),
            )
            .collect::<Vec<_>>();

        let n = self.history.len();
        let kernel_matrix = kernel_matrix(&self.kernel, &self.kernel_params, &v, &v).unwrap();

        let mut sum = 0.0;

        for i in 0..n {
            sum += kernel_matrix[0][i + 1].abs()
                / (kernel_matrix[0][0].sqrt() * kernel_matrix[i + 1][i + 1].sqrt());
        }

        Ok(sum / n as f64)
    }
}

impl<S, A, K> SampleableDistribution for GeneralizedKernelDensity<S, A, K>
where
    S: RandomVariable,
    A: RandomVariable,
    K: PositiveDefiniteKernel<Vec<f64>>,
{
    fn sample(
        &self,
        _theta: &Self::Condition,
        rng: &mut dyn rand::RngCore,
    ) -> Result<Self::Value, DistributionError> {
        let len = self.history.len();
        let n = rng.gen_range(0usize..=len) - 1usize;
        let result = self.history[n].1.clone();
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use crate::nonparametric::GeneralizedKernelDensity;
    use crate::*;
    use opensrdk_kernel_method::RBF;
    use rand::prelude::*;
    #[test]
    fn it_works() {
        let history = vec![(2.0, 1.0); 20];
        let kernel = RBF;
        let kernel_params = [0.5, 0.5];
        let model = GeneralizedKernelDensity::new(history, kernel, kernel_params.to_vec());

        let mut rng = StdRng::from_seed([1; 32]);

        let x = model.sample(&0.0, &mut rng).unwrap();

        println!("{:#?}", x);
    }
}
