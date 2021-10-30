extern crate blas_src;
extern crate lapack_src;
extern crate opensrdk_kernel_method;
extern crate opensrdk_linear_algebra;
extern crate opensrdk_probability;
extern crate plotters;
extern crate rayon;

use crate::distribution::Distribution;
use opensrdk_probability::nonparametric::*;
use opensrdk_probability::*;
use rand::prelude::*;
use rand::prelude::*;
use std::time::Instant;

#[test]
fn test_main() {}

fn it_works() {
    let mut rng = StdRng::from_seed([1; 32]);

    let mu1 = 1.0;
    let sigma1 = 103.0;
    let x1 = (0..100)
        .into_iter()
        .map(|_| {
            Normal
                .sample(&NormalParams::new(mu1, sigma1).unwrap(), &mut rng)
                .unwrap()
        })
        .collect::<Vec<_>>();

    let mu2 = 5.0;
    let sigma2 = 10.0;
    let x2 = (0..100)
        .into_iter()
        .map(|_| {
            Normal
                .sample(&NormalParams::new(mu2, sigma2).unwrap(), &mut rng)
                .unwrap()
        })
        .collect::<Vec<_>>();

    let x = [x1, x2].concat();
    let n = x.len();

    let alpha = 0.5;
    let d = 0.5;

    // let distr = InstantDistribution::new(
    //     &|x: &Vec<f64>, theta: &DirichletRandomMeasure<NormalParams>| {
    //         x.iter()
    //             .enumerate()
    //             .map(|(i, xi)| Normal.p(xi, &theta.w_theta[theta.z[i]].1))
    //             .product::<Result<f64, _>>()
    //     },
    //     &|theta, rng| {
    //         (0..theta.z.len())
    //             .into_iter()
    //             .map(|i| Normal.sample(&theta.w_theta[theta.z[i]].1, rng))
    //             .collect::<Result<Vec<_>, _>>()
    //     },
    // );

    // let g0 = BaselineMeasure::new(InstantDistribution::new(
    //     &|_: &NormalParams, _: &()| Ok(0.1),
    //     &|_, rng| {
    //         NormalParams::new(
    //             10.0 * rng.gen_range(0.0..=1.0),
    //             10.0 * rng.gen_range(0.0..=1.0) + 10.0,
    //         )
    //     },
    // ));
    // let pyp = StickBreakingProcess::<InstantDistribution<_, ()>, NormalParams>::new(4, n);
    // let pyp_params =
    //     PitmanYorProcessParams::new(DirichletProcessParams::new(alpha, g0).unwrap(), d).unwrap();
    // let condition = |_: &()| Ok(pyp_params.clone());

    // let pyp_conditioned = pyp.condition(&condition);

    // let mh_sampler = MetropolisHastingsSampler::new(&x, &distr, &pyp_conditioned, ());

    // let mut most_likely_result = (
    //     0.0,
    //     DirichletRandomMeasure::<NormalParams>::new(vec![], vec![]),
    // );

    // for _ in 0..20 {
    //     let g = mh_sampler
    //         .sample(5, pyp_conditioned.sample(&(), &mut rng).unwrap(), &mut rng)
    //         .unwrap();
    //     let p = distr.p(&x, &g).unwrap() * pyp_conditioned.p(&g, &()).unwrap();

    //     if most_likely_result.0 < p {
    //         most_likely_result = (p, g);
    //     }
    // }

    // println!("クラスタ数:{}", most_likely_result.1.z().len());
}
