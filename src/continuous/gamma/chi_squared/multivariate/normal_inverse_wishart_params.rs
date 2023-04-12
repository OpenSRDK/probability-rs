// use crate::{DistributionError, NormalInverseWishartError, RandomVariable};
// use opensrdk_linear_algebra::{pp::trf::PPTRF, SymmetricPackedMatrix};

// #[derive(Clone, Debug, PartialEq)]
// pub struct NormalInverseWishartParams {
//     mu0: Vec<f64>,
//     lambda: f64,
//     lpsi: PPTRF,
//     nu: f64,
// }

// impl NormalInverseWishartParams {
//     pub fn new(
//         mu0: Vec<f64>,
//         lambda: f64,
//         lpsi: PPTRF,
//         nu: f64,
//     ) -> Result<Self, DistributionError> {
//         let n = mu0.len();
//         if n != lpsi.0.dim() {
//             return Err(DistributionError::InvalidParameters(
//                 NormalInverseWishartError::DimensionMismatch.into(),
//             ));
//         }
//         if lambda <= 0.0 {
//             return Err(DistributionError::InvalidParameters(
//                 NormalInverseWishartError::DimensionMismatch.into(),
//             ));
//         }
//         if nu <= n as f64 - 1.0 {
//             return Err(DistributionError::InvalidParameters(
//                 NormalInverseWishartError::DimensionMismatch.into(),
//             ));
//         }

//         Ok(Self {
//             mu0,
//             lambda,
//             lpsi,
//             nu,
//         })
//     }

//     pub fn mu0(&self) -> &Vec<f64> {
//         &self.mu0
//     }

//     pub fn lambda(&self) -> f64 {
//         self.lambda
//     }

//     pub fn lpsi(&self) -> &PPTRF {
//         &self.lpsi
//     }

//     pub fn nu(&self) -> f64 {
//         self.nu
//     }
// }

// impl RandomVariable for NormalInverseWishartParams {
//     type RestoreInfo = usize;

//     fn transform_vec(&self) -> (Vec<f64>, Self::RestoreInfo) {
//         let n = self.mu0().len();
//         (
//             [
//                 self.mu0(),
//                 self.lpsi().0.elems(),
//                 &[self.lambda],
//                 &[self.nu],
//             ]
//             .concat(),
//             n,
//         )
//     }

//     fn len(&self) -> usize {
//         let t = self.lpsi.0.elems().len();
//         t + self.mu0.len() + 2usize
//     }

//     fn restore(v: &[f64], info: &Self::RestoreInfo) -> Result<Self, DistributionError> {
//         let n = *info;
//         let mu0 = v[0..n].to_vec();
//         let lpsi = PPTRF(SymmetricPackedMatrix::from(n, v[n..v.len() - 2].to_vec()).unwrap());
//         let lambda = v[v.len() - 2];
//         let nu = v[v.len() - 1];
//         Self::new(mu0, lambda, lpsi, nu)
//     }
// }
