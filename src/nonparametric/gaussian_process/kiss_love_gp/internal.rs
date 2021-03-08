use super::{grid::Grid, GaussianProcessError, GaussianProcessParams, KissLoveGP};
use crate::opensrdk_linear_algebra::*;
use opensrdk_kernel_method::{Convolutable, Kernel};
use rayon::prelude::*;
use std::{cmp::Ordering, error::Error};

impl<K, T> KissLoveGP<K, T>
where
    K: Kernel<Vec<f64>>,
    T: Convolutable + PartialEq,
{
    pub(crate) fn reset_prepare(&mut self) -> Result<&mut Self, Box<dyn Error>> {
        self.ready_to_predict = false;
        self.a = vec![];
        self.s = vec![];

        Ok(self)
    }

    pub(crate) fn wx(
        &self,
        x: &Vec<T>,
        force_with_u: bool,
    ) -> Result<(Vec<SparseMatrix>, Option<Grid>), Box<dyn Error>> {
        let n = x.len();
        if n == 0 {
            return Err(GaussianProcessError::Empty.into());
        }

        let parts_len = x[0].parts_len();
        let data_len = x[0].data_len();
        if parts_len == 0 || data_len == 0 {
            return Err(GaussianProcessError::Empty.into());
        }

        if force_with_u || self.u.axes().len() == 0 {
            let points = vec![(n / 2usize.pow(data_len as u32)).max(2); data_len];
            let u = Grid::from(&x, &points)?;

            let wx = self.u.interpolation_weight(&x)?;
            return Ok((wx, Some(u)));
        } else {
            let wx = self.u.interpolation_weight(&x)?;

            return Ok((wx, None));
        }
    }

    pub(crate) fn for_multivariate_normal(
        &self,
        params: &GaussianProcessParams<T>,
    ) -> Result<(KroneckerMatrices, Vec<SparseMatrix>), Box<dyn Error>> {
        let (wx, u) = if let Some(x) = params.x.as_ref() {
            self.wx(x, true)?
        } else {
            (self.wx.clone(), None)
        };
        let kuu = if let Some(u) = u {
            u.kuu(&self.kernel, &self.theta)?
        } else {
            self.u.kuu(&self.kernel, &self.theta)?
        };

        return Ok((kuu, wx));
    }

    /// See Andrew Gordon Wilson
    pub(crate) fn det_kxx(
        kuu: &KroneckerMatrices,
        wx: &Vec<SparseMatrix>,
    ) -> Result<f64, Box<dyn Error>> {
        let m = wx[0].rows;
        let n = wx[0].cols;

        let kuu_toeplitz = kuu
            .matrices()
            .iter()
            .map(|kp| Ok(ToeplitzMatrix::new(kp[0].to_vec(), kp[0][1..].to_vec())?))
            .collect::<Result<Vec<_>, Box<dyn Error>>>()?;

        let lambda = kuu_toeplitz
            .par_iter()
            .map(|kp| {
                kp.embedded_circulant().cigv().1[..kp.dim()]
                    .to_owned()
                    .col_mat()
            })
            .collect::<Vec<_>>();
        let lambda = KroneckerMatrices::new(lambda);
        let mut lambda = lambda.prod().vec();

        lambda.sort_by(|a, b| {
            a.re.partial_cmp(&b.re).unwrap_or(if !a.re.is_finite() {
                Ordering::Less
            } else {
                Ordering::Greater
            })
        });
        if !lambda[0].re.is_finite() {
            return Err(GaussianProcessError::NaNContamination.into());
        }

        let lambda = &lambda[m - n..];

        let det = lambda
            .par_iter()
            .map(|lmd| ((n as f64 / m as f64) * lmd.re))
            .product::<f64>();

        Ok(det)
    }

    pub(crate) fn luu(kuu: KroneckerMatrices) -> Result<KroneckerMatrices, Box<dyn Error>> {
        let matrices = kuu.eject();
        Ok(KroneckerMatrices::new(
            matrices
                .into_iter()
                .map(|kpi| kpi.potrf())
                .collect::<Result<Vec<_>, _>>()?,
        ))
    }
}
