use crate::{
    ConditionDifferentiableDistribution, ContinuousUniform, Distribution, DistributionError,
    RandomVariable, ValueDifferentiableDistribution,
};
use opensrdk_linear_algebra::*;
use rand::prelude::*;

/// Sample b from posterior p(b|a) with likelihood p(a|b) and prior p(b)
pub struct HamiltonianSampler<'a, L, P, A, B>
where
    L: Distribution<Value = A, Condition = B> + ConditionDifferentiableDistribution,
    P: Distribution<Value = B, Condition = ()> + ValueDifferentiableDistribution,
    A: RandomVariable,
    B: RandomVariable,
{
    value: &'a A,
    likelihood: &'a L,
    prior: &'a P,
}

#[derive(thiserror::Error, Debug)]
pub enum HamiltonianSamplingError {
    #[error("out of range")]
    OutOfRange,
    #[error("Unknown error")]
    Unknown,
}

impl<'a, L, P, A, B> HamiltonianSampler<'a, L, P, A, B>
where
    L: Distribution<Value = A, Condition = B> + ConditionDifferentiableDistribution,
    P: Distribution<Value = B, Condition = ()> + ValueDifferentiableDistribution,
    A: RandomVariable,
    B: RandomVariable,
{
    pub fn new(value: &'a A, likelihood: &'a L, prior: &'a P) -> Self {
        Self {
            value,
            likelihood,
            prior,
        }
    }

    pub fn sample(&self, p: Vec<f64>, rng: &mut dyn RngCore) -> Result<B, DistributionError> {
        // 事後分布のカーネル
        // (self.likelihood.fk(self.value, hoge)? * self.prior.fk(hoge, &())?).ln();
        // is ln probability density function kernel of posterior
        // 事後分布のValue微分
        // self.likelihood.ln_diff_condition(self.value, hoge)?.col_mat() + self.prior.ln_diff_value(hoge, &())?.col_mat();
        // is ln diff of posterior
        let mut b = self.prior.sample(&(), rng)?;
        let posterior_kernel = (self.likelihood.fk(self.value, &b)? * self.prior.fk(&b, &())?).ln();
        let posterior_diff = self.likelihood.ln_diff_condition(self.value, &b)?.col_mat()
            + self.prior.ln_diff_value(&b, &())?.col_mat();
        // h(theta)
        let posterior_potential = -posterior_kernel.ln();
        // dh_dtheta(theta)
        let entropy_diff = -posterior_diff;
        // h(theta)+1/2*p^2
        let prev_hamiltonian =
            posterior_potential + 0.5 * p.clone().col_mat().hadamard_prod(&p.clone().col_mat());
        println!("{:#?}", prev_hamiltonian);
        const L: usize = 100;
        const T: usize = 10000;
        let eps = 0.01;
        let kinetic_p = p.col_mat();
        let kinetic_theta = self.value.clone().transform_vec();

        for _t in 0..T {
            for _i in 0..L {
                // 0.5ステップ後のp
                // 1ステップ後のtheta
                // 1ステップ後のp
            }
            // let hamiltonian = hamiltonian(kinetic_p, kinetic_theta)?;
            // let r = (prev_hamiltonian - hamiltonian).exp();
            // if r > 1.0 {
            //     sim_result.append(&mut vec![(kinetic_p, kinetic_theta)]);
            //     sim_result_hamiltonian.append(&mut vec![prev_hamiltonian]);
            // } else if r > 0.0 && r > ContinuousUniform.sample(&(0.0..1.0), rng)? {
            //     sim_result.append(&mut vec![(kinetic_p, kinetic_theta)]);
            //     sim_result_hamiltonian.append(&mut vec![prev_hamiltonian]);
            // }
        }
        Ok()
    }

    fn hamiltonian(p: f64, x: f64) -> Result<f64, DistributionError> {
        let lambda = 13.0;
        let alpha = 11.0;
        Ok(lambda * x - (alpha - 1.0) * x.ln() + 0.5 * p.powf(2.0))
    }

    fn leapfrog_next_half_p(p: &[f64], x: &[f64]) -> Result<f64, DistributionError> {
        let lambda = 13.0;
        let alpha = 11.0;
        let eps = 0.01;
        Ok(p - 0.5 * eps * (lambda - (alpha - 1.0) / x))
    }

    fn leapfrog_next_theta(p: f64, x: f64) -> Result<f64, DistributionError> {
        let eps = 0.01;
        Ok(x + eps * p)
    }
}

fn hamiltonian(p: f64, theta: f64) -> Result<f64, DistributionError> {
    let lambda = 13.0;
    let alpha = 11.0;
    Ok(lambda * theta - (alpha - 1.0) * theta.ln() + 0.5 * p.powf(2.0))
}

fn leapfrog_next_half_p(p: f64, theta: f64) -> Result<f64, DistributionError> {
    let lambda = 13.0;
    let alpha = 11.0;
    let eps = 0.01;
    Ok(p - 0.5 * eps * (lambda - (alpha - 1.0) / theta))
}

fn leapfrog_next_theta(p: f64, theta: f64) -> Result<f64, DistributionError> {
    let eps = 0.01;
    Ok(theta + eps * p)
}

// r=exp⁡(H(θ(t),p(t))−H(θ(a),p(a)))
// これと標準正規分布を比較して受容、非受容を決定
fn sample(
    theta: f64, //D::Condition,
    rng: &mut dyn RngCore,
) -> Result<Vec<(f64, f64)>, DistributionError> {
    const L: usize = 100;
    const T: usize = 10000;
    let mut kinetic_p = 0.0;
    let mut kinetic_theta = theta;
    let mut sim_result = vec![];
    let mut sim_result_hamiltonian = vec![];
    let prev_hamiltonian = hamiltonian(kinetic_p, theta)?;
    println!("{}", prev_hamiltonian);
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
            sim_result.append(&mut vec![(kinetic_p, kinetic_theta)]);
            sim_result_hamiltonian.append(&mut vec![prev_hamiltonian]);
        } else if r > 0.0 && r > ContinuousUniform.sample(&(0.0..1.0), rng)? {
            sim_result.append(&mut vec![(kinetic_p, kinetic_theta)]);
            sim_result_hamiltonian.append(&mut vec![prev_hamiltonian]);
        }
    }
    Ok(sim_result)
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
    use crate::hamiltonian::sample;
    use plotters::prelude::*;
    use rand::{prelude::StdRng, SeedableRng};

    #[test]
    fn it_works() -> Result<(), Box<dyn std::error::Error>> {
        let theta = 2.5;
        let mut rng = StdRng::from_seed([1; 32]);
        let hamiltonian = sample(theta, &mut rng).unwrap();
        let xs: Vec<f64> = hamiltonian.iter().map(|(x, _)| *x).collect();
        let ys: Vec<f64> = hamiltonian.iter().map(|(_, y)| *y).collect();

        let image_width = 1080;
        let image_height = 720;
        let root =
            BitMapBackend::new("hamiltonian.png", (image_width, image_height)).into_drawing_area();
        root.fill(&WHITE)?;
        let (x_min, x_max) = xs
            .iter()
            .fold((0.0 / 0.0, 0.0 / 0.0), |(m, n), v| (v.min(m), v.max(n)));
        let (y_min, y_max) = ys
            .iter()
            .fold((0.0 / 0.0, 0.0 / 0.0), |(m, n), v| (v.min(m), v.max(n)));
        let caption = "Hamiltonian Sampling";
        let font = ("sans-serif", 20);
        let mut chart = ChartBuilder::on(&root)
            .caption(caption, font.into_font())
            .margin(10)
            .x_label_area_size(16)
            .y_label_area_size(42)
            .build_cartesian_2d(x_min..x_max, y_min..y_max)?;
        chart.configure_mesh().draw()?;

        let line_series = LineSeries::new(xs.iter().zip(ys.iter()).map(|(x, y)| (*x, *y)), &RED);
        chart.draw_series(line_series)?;

        Ok(())
    }
}
