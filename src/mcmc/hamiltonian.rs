use crate::{ContinuousUniform, Distribution, DistributionError, RandomVariable};
use rand::prelude::*;

/// Sample b from posterior p(b|a) with likelihood p(a|b) and prior p(b)
pub struct HamiltonianSampler<T, D>
where
    T: RandomVariable,
    D: Distribution<Value = T, Condition = ()>,
{
    distribution: D,
}

#[derive(thiserror::Error, Debug)]
pub enum ImportanceSamplingError {
    #[error("out of range")]
    OutOfRange,
    #[error("Unknown error")]
    Unknown,
}

impl<T, D> HamiltonianSampler<T, D>
where
    T: RandomVariable,
    D: Distribution<Value = T, Condition = ()>,
{
    pub fn new(distribution: D) -> Result<Self, DistributionError> {
        Ok(Self { distribution })
    }

    // r=exp⁡(H(θ(t),p(t))−H(θ(a),p(a)))
    // これと標準正規分布を比較して受容、非受容を決定
    pub fn sample(
        &self,
        theta: f64, //D::Condition,
        rng: &mut dyn RngCore,
    ) -> Result<Vec<f64>, DistributionError> {
        const L: usize = 100;
        const T: usize = 10000;
        let mut kinetic_p = 0.0;
        let mut kinetic_theta = theta;
        let mut sim_result_hamiltonian = vec![];
        let prev_hamiltonian = hamiltonian(kinetic_p, theta)?;
        sim_result_hamiltonian.append(&mut vec![prev_hamiltonian]);
        for _t in 0..T {
            for _i in 0..L {
                kinetic_p = leapfrog_next_half_p(kinetic_p, kinetic_theta)?;
                kinetic_theta = leapfrog_next_theta(kinetic_p, kinetic_theta)?;
                kinetic_p = leapfrog_next_half_p(kinetic_p, kinetic_theta)?;
            }
            let hamiltonian = hamiltonian(kinetic_p, kinetic_theta)?;
            let r = (prev_hamiltonian - hamiltonian).exp();
            if r > 1.0 {
                sim_result_hamiltonian.append(&mut vec![prev_hamiltonian]);
            } else if r > 0.0 && r > ContinuousUniform.sample(&(0.0..1.0), rng)? {
                sim_result_hamiltonian.append(&mut vec![prev_hamiltonian]);
            }
        }
        Ok(sim_result_hamiltonian)
    }
}

fn hamiltonian(p: f64, theta: f64) -> Result<f64, DistributionError> {
    let lambda = 1.0;
    let alpha = 1.0;
    Ok(lambda * theta - (alpha - 1.0) * theta.ln() + 0.5 * p.powf(2.0))
}

fn leapfrog_next_half_p(p: f64, theta: f64) -> Result<f64, DistributionError> {
    let lambda = 1.0;
    let alpha = 1.0;
    let eps = 0.01;
    Ok(p - 0.5 * eps * (lambda - (alpha - 1.0) / theta))
}

fn leapfrog_next_theta(p: f64, theta: f64) -> Result<f64, DistributionError> {
    let eps = 0.01;
    Ok(theta + eps * p)
}

// fn move_one_step(p: f64, theta: f64) -> Result<Vec<f64>, DistributionError> {
//     const L: usize = 100;
//     let mut kinetic_p = p;
//     let mut kinetic_theta = theta;
//     let mut p_sample_hamiltonian = vec![];
//     p_sample_hamiltonian.append(&mut vec![hamiltonian(p, theta)?]);
//     for _i in 0..L {
//         kinetic_p = leapfrog_next_half_p(kinetic_p, kinetic_theta)?;
//         kinetic_theta = leapfrog_next_theta(kinetic_p, kinetic_theta)?;
//         kinetic_p = leapfrog_next_half_p(kinetic_p, kinetic_theta)?;
//         p_sample_hamiltonian.append(&mut vec![hamiltonian(p, theta)?])
//     }
//     Ok(p_sample_hamiltonian)
// }

#[cfg(test)]
mod test {
    use crate::{HamiltonianSampler, Normal};

    #[test]
    fn it_works() {
        let theta = 2.5;
        let sample = HamiltonianSampler::new(Normal).sample();
    }
}
