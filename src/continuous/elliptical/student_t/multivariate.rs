use crate::{ContinuousDistribution, JointDistribution};
use opensrdk_symbolic_computation::{Expression, Size};
use serde::{Deserialize, Serialize};
use std::{collections::HashSet, ops::Mul};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MultivariateStudentT {
    x: Expression,
    nu: Expression,
}

impl MultivariateStudentT {
    pub fn new(x: Expression, nu: Expression) -> MultivariateStudentT {
        if x.mathematical_sizes() != vec![Size::Many, Size::One] && x.mathematical_sizes() != vec![]
        {
            panic!("x must be a scalar or a 2 rank vector");
        }
        MultivariateStudentT { x, nu }
    }
}

impl<Rhs> Mul<Rhs> for MultivariateStudentT
where
    Rhs: ContinuousDistribution,
{
    type Output = JointDistribution<Self, Rhs>;

    fn mul(self, rhs: Rhs) -> Self::Output {
        JointDistribution::new(self, rhs)
    }
}

impl ContinuousDistribution for MultivariateStudentT {
    fn value_ids(&self) -> HashSet<&str> {
        self.x.variable_ids()
    }

    fn conditions(&self) -> Vec<&Expression> {
        vec![&self.nu]
    }

    fn pdf(&self) -> Expression {
        let x = self.x.clone();
        let nu = self.nu.clone();

        let pdf_expression = todo!();

        pdf_expression
    }

    fn condition_ids(&self) -> HashSet<&str> {
        self.conditions()
            .iter()
            .map(|v| v.variable_ids())
            .flatten()
            .collect::<HashSet<_>>()
            .difference(&self.value_ids())
            .cloned()
            .collect()
    }

    fn ln_pdf(&self) -> Expression {
        self.pdf().ln()
    }
}

// #[derive(thiserror::Error, Debug)]
// pub enum MultivariateStudentTError {
//     #[error("dimension mismatch (StudentT Multivariate)")]
//     DimensionMismatch,
// }

// impl<T, U> Distribution for MultivariateStudentT<T, U>
// where
//     T: MultivariateStudentTParams<U>,
//     U: EllipticalParams,
// {
//     type Value = Vec<f64>;
//     type Condition = T;

//     fn p_kernel(&self, x: &Self::Value, theta: &Self::Condition) -> Result<f64, DistributionError> {
//         let elliptical = theta.elliptical();
//         let x_mu = elliptical.x_mu(x)?.col_mat();

//         let n = x_mu.rows() as f64;
//         let nu = theta.nu();

//         Ok((1.0 + (x_mu.t() * elliptical.sigma_inv_mul(x_mu)?)[(0, 0)] / nu).powf(-(nu + n) / 2.0))
//     }
// }

// impl ValueDifferentiableDistribution for MultivariateStudentT {
//     fn ln_diff_value(
//         &self,
//         x: &Self::Value,
//         theta: &Self::Condition,
//     ) -> Result<Vec<f64>, DistributionError> {
//         let x_mat = x.clone().row_mat();
//         let mu_mat = theta.mu().clone().row_mat();
//         let x_mu = x_mat - mu_mat;
//         let x_mu_t = x_mu.t();
//         let sigma_inv = theta.lsigma().clone().pptri()?.to_mat();
//         let nu = theta.nu();
//         let n = x.len() as f64;
//         let d = (&x_mu * &sigma_inv * &x_mu_t)[(0, 0)];
//         let f_x = -(&nu + &n) / &nu * (1.0 + &d).powi(-1) * (x_mu * sigma_inv);
//         Ok(f_x.vec())
//     }
// }

// impl ConditionDifferentiableDistribution for MultivariateStudentT {
//     fn ln_diff_condition(
//         &self,
//         x: &Self::Value,
//         theta: &Self::Condition,
//     ) -> Result<Vec<f64>, DistributionError> {
//         let x_mat = x.clone().row_mat();
//         let mu_mat = theta.mu().clone().row_mat();
//         let x_mu = x_mat - mu_mat;
//         let x_mu_t = x_mu.t();
//         let sigma_inv = theta.lsigma().clone().pptri()?.to_mat();
//         let nu = theta.nu();
//         let n = x.len() as f64;
//         let d = (&x_mu * &sigma_inv * &x_mu_t)[(0, 0)];
//         // Hadamard product (L*L*L)
//         let m = sigma_inv
//             .clone()
//             .hadamard_prod(&sigma_inv)
//             .hadamard_prod(&sigma_inv);
//         let f_mu = (&nu + &n) / &nu * (1.0 + &d).powi(-1) * (&x_mu * &sigma_inv);
//         let f_lsigma = (&nu + &n) / &nu * (1.0 + &d / &nu).powi(-1) * (&x_mu * &m * &x_mu_t);
//         let f_nu = 0.5
//             * ((0.5 * (nu + n)).digamma()
//                 - (n / nu)
//                 - (0.5 * nu).digamma()
//                 - (nu + n) * d / nu.powi(2) * (1.0 + d / nu).powi(-1)
//                 - (1.0 + d / nu).ln());
//         Ok([f_mu.vec(), f_lsigma.vec(), vec![f_nu]].concat())
//     }
// }

// pub trait MultivariateStudentTParams<T>: RandomVariable
// where
//     T: EllipticalParams,
// {
//     fn nu(&self) -> f64;
//     fn elliptical(&self) -> &T;
// }

// #[derive(Clone, Debug)]
// pub struct ExactMultivariateStudentTParams {
//     nu: f64,
//     elliptical: ExactEllipticalParams,
// }

// impl ExactMultivariateStudentTParams {
//     /// # Multivariate student t
//     /// `L` is needed as second argument under decomposition `Sigma = L * L^T`
//     /// lsigma = sigma.potrf()?;
//     pub fn new(nu: f64, mu: Vec<f64>, lsigma: PPTRF) -> Result<Self, DistributionError> {
//         let elliptical = ExactEllipticalParams::new(mu, lsigma)?;

//         Ok(Self { nu, elliptical })
//     }

//     pub fn mu(&self) -> &Vec<f64> {
//         self.elliptical.mu()
//     }

//     pub fn lsigma(&self) -> &PPTRF {
//         self.elliptical.lsigma()
//     }
// }

// impl RandomVariable for ExactMultivariateStudentTParams {
//     type RestoreInfo = usize;

//     fn transform_vec(&self) -> (Vec<f64>, Self::RestoreInfo) {
//         let p = self.mu().len();
//         // let p = self.lsigma().0.dim();
//         ([self.mu(), self.lsigma().0.elems(), &[self.nu]].concat(), p)
//     }

//     fn len(&self) -> usize {
//         let t = self.elliptical.lsigma().0.elems().len();
//         t + self.elliptical.mu().len() + 1usize
//     }

//     fn restore(v: &[f64], info: &Self::RestoreInfo) -> Result<Self, DistributionError> {
//         let p = *info;
//         let mu = v[0..p].to_vec();
//         let lsigma = PPTRF(SymmetricPackedMatrix::from(p, v[p..v.len() - 1].to_vec()).unwrap());
//         let nu = v[v.len() - 1];
//         Self::new(nu, mu, lsigma)
//     }
// }

// impl MultivariateStudentTParams<ExactEllipticalParams> for ExactMultivariateStudentTParams {
//     fn nu(&self) -> f64 {
//         self.nu
//     }

//     fn elliptical(&self) -> &ExactEllipticalParams {
//         &self.elliptical
//     }
// }

// impl<T, U, Rhs, TRhs> Mul<Rhs> for MultivariateStudentT<T, U>
// where
//     T: MultivariateStudentTParams<U>,
//     U: EllipticalParams,
//     Rhs: Distribution<Value = TRhs, Condition = T>,
//     TRhs: RandomVariable,
// {
//     type Output = IndependentJoint<Self, Rhs, Vec<f64>, TRhs, T>;

//     fn mul(self, rhs: Rhs) -> Self::Output {
//         IndependentJoint::new(self, rhs)
//     }
// }

// impl<T, U, Rhs, URhs> BitAnd<Rhs> for MultivariateStudentT<T, U>
// where
//     T: MultivariateStudentTParams<U>,
//     U: EllipticalParams,
//     Rhs: Distribution<Value = T, Condition = URhs>,
//     URhs: RandomVariable,
// {
//     type Output = DependentJoint<Self, Rhs, Vec<f64>, T, URhs>;

//     fn bitand(self, rhs: Rhs) -> Self::Output {
//         DependentJoint::new(self, rhs)
//     }
// }

// impl SamplableDistribution for MultivariateStudentT {
//     fn sample(
//         &self,
//         theta: &Self::Condition,
//         rng: &mut dyn RngCore,
//     ) -> Result<Self::Value, DistributionError> {
//         let nu = theta.nu();
//         let elliptical = theta.elliptical();

//         let student_t = match RandStudentT::new(nu) {
//             Ok(v) => Ok(v),
//             Err(e) => Err(DistributionError::Others(e.into())),
//         }?;

//         let z = (0..elliptical.lsigma_cols())
//             .into_iter()
//             .map(|_| rng.sample(student_t))
//             .collect::<Vec<_>>();

//         Ok(elliptical.sample(z)?)
//     }
// }

// #[cfg(test)]
// mod tests {
//     use crate::{
//         ConditionDifferentiableDistribution, Distribution, ExactMultivariateStudentTParams,
//         MultivariateStudentT, SamplableDistribution, ValueDifferentiableDistribution,
//     };
//     use opensrdk_linear_algebra::{pp::trf::PPTRF, *};
//     use rand::prelude::*;
//     #[test]
//     fn it_works() {
//         let student_t = MultivariateStudentT::new();
//         let mut rng = StdRng::from_seed([1; 32]);

//         let mu = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
//         let lsigma = SymmetricPackedMatrix::from_mat(&mat!(
//            1.0,  0.0,  0.0,  0.0,  0.0,  0.0;
//            2.0,  3.0,  0.0,  0.0,  0.0,  0.0;
//            4.0,  5.0,  6.0,  0.0,  0.0,  0.0;
//            7.0,  8.0,  9.0, 10.0,  0.0,  0.0;
//           11.0, 12.0, 13.0, 14.0, 15.0,  0.0;
//           16.0, 17.0, 18.0, 19.0, 20.0, 21.0
//         ))
//         .unwrap();
//         println!("{:#?}", lsigma);

//         let x = student_t
//             .sample(
//                 &ExactMultivariateStudentTParams::new(1.0, mu, PPTRF(lsigma)).unwrap(),
//                 &mut rng,
//             )
//             .unwrap();

//         println!("{:#?}", x);
//     }

//     #[test]
//     fn it_works2() {
//         let student_t = MultivariateStudentT::new();

//         let mu = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
//         let lsigma = SymmetricPackedMatrix::from_mat(&mat!(
//            1.0,  0.0,  0.0,  0.0,  0.0,  0.0;
//            2.0,  3.0,  0.0,  0.0,  0.0,  0.0;
//            4.0,  5.0,  6.0,  0.0,  0.0,  0.0;
//            7.0,  8.0,  9.0, 10.0,  0.0,  0.0;
//           11.0, 12.0, 13.0, 14.0, 15.0,  0.0;
//           16.0, 17.0, 18.0, 19.0, 20.0, 21.0
//         ))
//         .unwrap();
//         println!("{:#?}", lsigma);

//         let x = vec![0.0, 1.0, 2.0, 0.0, 1.0, 2.0];

//         let f = student_t.ln_diff_value(
//             &x,
//             &ExactMultivariateStudentTParams::new(1.0, mu, PPTRF(lsigma)).unwrap(),
//         );
//         println!("{:#?}", f);
//     }

//     #[test]
//     fn it_works_3() {
//         let student_t = MultivariateStudentT::new();

//         let mu = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
//         let lsigma = SymmetricPackedMatrix::from_mat(&mat!(
//            1.0,  0.0,  0.0,  0.0,  0.0,  0.0;
//            2.0,  3.0,  0.0,  0.0,  0.0,  0.0;
//            4.0,  5.0,  6.0,  0.0,  0.0,  0.0;
//            7.0,  8.0,  9.0, 10.0,  0.0,  0.0;
//           11.0, 12.0, 13.0, 14.0, 15.0,  0.0;
//           16.0, 17.0, 18.0, 19.0, 20.0, 21.0
//         ))
//         .unwrap();
//         println!("{:#?}", lsigma);

//         let x = vec![0.0, 1.0, 2.0, 0.0, 1.0, 2.0];

//         let f = student_t.ln_diff_condition(
//             &x,
//             &ExactMultivariateStudentTParams::new(1.0, mu, PPTRF(lsigma)).unwrap(),
//         );
//         println!("{:#?}", f);
//     }
// }
