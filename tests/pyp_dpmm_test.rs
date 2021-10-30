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
use plotters::prelude::*;
use rand::prelude::*;
use std::collections::{HashMap, HashSet};
use std::time::Instant;

#[test]
fn test_main() {
    let is_not_ci = true;

    if is_not_ci {
        let start = Instant::now();

        it_works().unwrap();

        let end = start.elapsed();
        println!("{}.{:03}sec", end.as_secs(), end.subsec_nanos() / 1_000_000);
    }
}

fn it_works() -> Result<(), Box<dyn std::error::Error>> {
    let root = BitMapBackend::gif("dpmm.gif", (1600, 900), 0_500)?.into_drawing_area();

    let mut rng = StdRng::from_seed([1; 32]);

    let x = (0..=4)
        .into_iter()
        .flat_map(|i| {
            let mu = 10.0 * i as f64 - 20.0;
            let theta = NormalParams::new(mu, 1.0).unwrap();
            let mut x = vec![0.0; 100];
            for i in 0..100 {
                x[i] = Normal.sample(&theta, &mut rng).unwrap()
            }
            x.into_iter()
        })
        .collect::<Vec<_>>();

    println!("x生成完了");

    let n = x.len();
    let mut s = (1u32..=n as u32).into_iter().collect::<Vec<_>>();
    let mut theta = (1..=n as u32)
        .into_iter()
        .map(|k| (k, NormalParams::new(0.0, 1.0).unwrap()))
        .collect::<HashMap<u32, NormalParams>>();
    let mut max_k = theta.len() as u32;

    let alpha = 0.5;
    let d = 0.5;
    let pyp_params = PitmanYorProcessParams::new(alpha, d)?;

    let g0 = BaselineMeasure::new(InstantDistribution::new(
        &|_: &NormalParams, _: &()| Ok(0.1),
        &|_, rng| NormalParams::new(20.0 * rng.gen_range(-1.0..=1.0), 10.0),
    ));

    let mh_proposal = InstantDistribution::new(
        &|x: &NormalParams, theta: &NormalParams| {
            Ok(Normal.p(&x.mu(), &NormalParams::new(theta.mu(), 10.0)?)?)
        },
        &|theta, rng| {
            let mu = Normal.sample(&NormalParams::new(theta.mu(), 10.0)?, rng)?;
            NormalParams::new(mu, 10.0)
        },
    );

    const ITER: usize = 40;

    for iter in 0..ITER {
        root.fill(&WHITE)?;
        let mut chart = ChartBuilder::on(&root)
            .margin(10)
            .set_all_label_area_size(50)
            .build_cartesian_2d(-30.0..30.0, -10.0..10.0)?;

        chart
            .configure_mesh()
            .x_labels(10)
            .y_labels(10)
            .disable_mesh()
            .x_label_formatter(&|v| format!("{:.1}", v))
            .y_label_formatter(&|v| format!("{:.1}", v))
            .draw()?;

        println!("iteration {}", iter);

        chart.draw_series(PointSeries::of_element(
            x.iter().map(|&xi| (xi, 0.0)).into_iter(),
            2,
            ShapeStyle::from(&BLACK.mix(0.1)).filled(),
            &|coord, size, style| EmptyElement::at(coord) + Circle::new((0, 0), size, style),
        ))?;

        for (i, &xi) in x.iter().enumerate() {
            let s_clone = s.clone();
            let condition = pyp_params.gibbs_condition(&s_clone, i);
            let gibbs_likelihood = Normal.switch(&theta, NormalParams::default());
            let gibbs_prior = PitmanYorGibbs::new().condition(&condition);

            let ds_sampler = DiscreteSliceSampler::new(
                &xi,
                &gibbs_likelihood,
                &gibbs_prior,
                &s.iter()
                    .map(|&si| si)
                    .chain(std::iter::once(0))
                    .collect::<HashSet<u32>>(),
            )?;

            let new_k = ds_sampler.sample(3, None, &mut rng)?;
            if new_k == 0 {
                s[i] = max_k + 1;
                max_k += 1;
            } else {
                s[i] = new_k;
            }
        }

        let mut theta_update_map = HashMap::new();
        let mut theta_remove_set = HashSet::new();
        for (&k, theta_k_init) in theta.iter() {
            let x_in_k = PitmanYorProcessParams::x_in_cluster(&x, &s, k);

            let len = x_in_k.len();
            if len == 0 {
                theta_remove_set.insert(k);
                continue;
            }

            let x_likelihood = vec![Normal; len].into_iter().joint();
            let mh_sampler =
                MetropolisHastingsSampler::new(&x_in_k, &x_likelihood, &g0.distr, &mh_proposal);

            let theta_k = mh_sampler.sample(4, theta_k_init.clone(), &mut rng)?;

            theta_update_map.insert(k, theta_k);
        }

        for (k, theta_k) in theta_update_map {
            theta.insert(k, theta_k);
        }

        for k in theta_remove_set {
            theta.remove(&k);
        }

        chart.draw_series(PointSeries::of_element(
            theta
                .iter()
                .map(|(_, theta_k)| (theta_k.mu(), 0.0))
                .into_iter(),
            10,
            ShapeStyle::from(&BLUE.mix(0.5)).stroke_width(1),
            &|coord, size, style| EmptyElement::at(coord) + Circle::new((0, 0), size, style),
        ))?;

        root.present()?;
    }

    theta.into_iter().enumerate().for_each(|(i, (k, v))| {
        let x_in_k = PitmanYorProcessParams::x_in_cluster(&x, &s, k);

        println!("クラスタ{}", i + 1);
        println!("\t所属データ数: {}", x_in_k.len());
        println!("\tμ: {}, σ: {}", v.mu(), v.sigma());
        println!("\t{:?}", x_in_k);
        println!("");
    });

    Ok(())
}
