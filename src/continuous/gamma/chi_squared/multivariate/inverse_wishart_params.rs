// use opensrdk_linear_algebra::{pp::trf::PPTRF, SymmetricPackedMatrix};

// use crate::{DistributionError, InverseWishartError, RandomVariable};

// #[derive(Clone, Debug, PartialEq)]
// pub struct InverseWishartParams {
//     lpsi: PPTRF,
//     nu: f64,
// }

// impl InverseWishartParams {
//     pub fn new(lpsi: PPTRF, nu: f64) -> Result<Self, DistributionError> {
//         let p = lpsi.0.dim();

//         if nu <= p as f64 - 1.0 {
//             return Err(DistributionError::InvalidParameters(
//                 InverseWishartError::NuMustBeGTEDimension.into(),
//             ));
//         }

//         Ok(Self { lpsi, nu })
//     }

//     pub fn lpsi(&self) -> &PPTRF {
//         &self.lpsi
//     }

//     pub fn nu(&self) -> f64 {
//         self.nu
//     }
// }

// impl RandomVariable for InverseWishartParams {
//     type RestoreInfo = usize;

//     fn transform_vec(&self) -> (Vec<f64>, Self::RestoreInfo) {
//         let p = self.lpsi.0.dim();
//         ([self.lpsi.0.elems(), &[self.nu]].concat(), p)
//     }

//     fn len(&self) -> usize {
//         let t = self.lpsi.0.elems().len();
//         t + 1usize
//     }

//     fn restore(v: &[f64], info: &Self::RestoreInfo) -> Result<Self, DistributionError> {
//         if v.len() != info + 1 {
//             return Err(DistributionError::InvalidRestoreVector);
//         }
//         let p = *info;
//         let nu = v[v.len() - 1];
//         let lpsi = PPTRF(SymmetricPackedMatrix::from(p, v[0..p].to_vec()).unwrap());
//         Self::new(lpsi, nu)
//     }
// }
