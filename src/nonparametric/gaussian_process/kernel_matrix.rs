use crate::opensrdk_linear_algebra::*;
use opensrdk_kernel_method::*;
use std::{error::Error, fmt::Debug};

pub fn kernel_matrix<T>(
    kernel: &impl Kernel<T>,
    params: &[f64],
    x: &[T],
    x_prime: &[T],
) -> Result<Matrix, Box<dyn Error>>
where
    T: Clone + Debug,
{
    let m = x.len();
    let n = x_prime.len();

    let mut k = Matrix::new(m, n);

    for i in 0..m {
        for j in 0..n {
            k[i][j] = kernel.value(params, &x[i], &x_prime[j], false)?.0;
        }
    }

    Ok(k)
}
