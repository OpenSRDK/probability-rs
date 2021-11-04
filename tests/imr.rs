extern crate blas_src;
extern crate lapack_src;
extern crate opensrdk_kernel_method;
extern crate opensrdk_linear_algebra;
extern crate opensrdk_probability;
extern crate plotters;
extern crate rayon;

use crate::distribution::Distribution;
use opensrdk_linear_algebra::*;
use opensrdk_probability::nonparametric::*;
use opensrdk_probability::*;
use plotters::prelude::*;
use rand::prelude::*;
// use rayon::prelude::*;
use std::collections::{HashMap, LinkedList};
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
    let mut rng = StdRng::from_seed([1; 32]);

    let x = (0..100).into_iter().collect::<Vec<_>>();
    let alpha_beta = vec![(40.0, -1.0), (5.0, 2.0), (65.0, -1.0)];
    let sigma = 2.0;
    let y_x = x
        .into_iter()
        .map(|xt| -> Result<(f64, f64), DistributionError> {
            let eps = Normal.sample(&NormalParams::new(0.0, sigma)?, &mut rng)?;
            let (alpha, beta) = if xt < 35 {
                alpha_beta[0]
            } else if xt < 65 {
                alpha_beta[1]
            } else {
                alpha_beta[2]
            };
            Ok((alpha + beta * xt as f64 + eps, xt as f64))
        })
        .collect::<Result<Vec<_>, DistributionError>>()?;

    println!("x生成完了");

    let n = x.len();

    let alpha_pyp = 0.7;
    let d = 0.1;

    let g0 = BaselineMeasure::new(InstantDistribution::new(
        &|_: _, _: &()| Ok(0.1),
        &|_, rng| {
            Ok((
                30.0 * rng.gen_range(-1.0..=1.0),
                30.0 * rng.gen_range(-1.0..=1.0),
            ))
        },
    ));

    let pyp_params = PitmanYorProcessParams::new(alpha_pyp, d, g0.clone())?;

    let mh_proposal = InstantDistribution::new(
        &|x: &(f64, f64), theta: &(f64, f64)| {
            Ok(MultivariateNormal::new().fk(
                &vec![x.0, x.1],
                &ExactEllipticalParams::new(
                    vec![theta.0, theta.1],
                    (3.0 * DiagonalMatrix::identity(2)).mat(),
                )?,
            )?)
        },
        &|theta, rng| {
            let x = MultivariateNormal::new().sample(
                &ExactEllipticalParams::new(
                    vec![theta.0, theta.1].clone(),
                    (3.0 * DiagonalMatrix::identity(2)).mat(),
                )?,
                rng,
            )?;
            Ok((x[0], x[1]))
        },
    );

    const ITER: usize = 2000;
    const BURNIN: usize = 1000;

    let mut state_list = LinkedList::<ClusterSwitch<(f64, f64)>>::new();
    state_list.push_back(ClusterSwitch::new(
        (0..n as u32).into_iter().collect::<Vec<_>>(),
        (0..n as u32)
            .into_iter()
            .map(|k| (k, (0.0, 0.0)))
            .collect::<HashMap<u32, (f64, f64)>>(),
    )?);

    let likelihood = InstantDistribution::new(
        &|x: &(f64, f64), theta: &(f64, f64)| {
            let (y, x) = *x;
            let (alpha, beta) = *theta;
            Normal.fk(&y, &NormalParams::new(alpha + beta * x, sigma)?)
        },
        &|theta: &(f64, f64), rng| Ok((0.0, 0.0)),
    );

    for iter in 0..ITER {
        println!("iteration {}", iter);

        let mut s = state_list.back().unwrap().clone();

        {
            let mut gibbs_sampler = PitmanYorGibbsSampler::<
                InstantDistribution<(f64, f64), (f64, f64)>,
                (f64, f64),
                (f64, f64),
                InstantDistribution<(f64, f64), ()>,
            >::new(&pyp_params, &mut s, &y_x, &likelihood);

            gibbs_sampler.sample(&mh_proposal, &mut rng)?;
        };

        if iter <= BURNIN {
            state_list.clear();
        }
        state_list.push_back(s);
        println!("{:?}", state_list.back().unwrap().theta().len());
    }

    // let mut max_p = 0.0;
    // let mut max_p_state = state_list.front().unwrap().clone();

    // let root = BitMapBackend::gif("dpmm.gif", (1600, 900), 0_500)?.into_drawing_area();

    // for (t, s_t) in state_list.into_iter().enumerate() {
    //     println!("gif {} writing...", t);

    //     let p = (0..n)
    //         .into_iter()
    //         .map(|i| MultivariateNormal::new().p(&x[i], s_t.theta().get(&s_t.s()[i]).unwrap()))
    //         .product::<Result<f64, DistributionError>>()?;

    //     if max_p < p || true {
    //         max_p = p;
    //         max_p_state = s_t;
    //     }
    //     println!("{}", max_p);

    //     root.fill(&WHITE)?;
    //     let mut chart = ChartBuilder::on(&root)
    //         .margin(10)
    //         .set_all_label_area_size(50)
    //         .build_cartesian_2d(-30.0..30.0, -30.0..30.0)?;

    //     chart
    //         .configure_mesh()
    //         .x_labels(10)
    //         .y_labels(10)
    //         .disable_mesh()
    //         .x_label_formatter(&|v| format!("{:.1}", v))
    //         .y_label_formatter(&|v| format!("{:.1}", v))
    //         .draw()?;

    //     chart.draw_series(PointSeries::of_element(
    //         x.iter().map(|xi| (xi[0], xi[1])).into_iter(),
    //         2,
    //         ShapeStyle::from(&BLACK.mix(0.1)).filled(),
    //         &|coord, size, style| EmptyElement::at(coord) + Circle::new((0, 0), size, style),
    //     ))?;

    //     chart.draw_series(PointSeries::of_element(
    //         max_p_state
    //             .s_inv()
    //             .iter()
    //             .map(|(k, _)| max_p_state.theta().get(k).unwrap())
    //             .map(|theta_k| (theta_k.mu()[0], theta_k.mu()[1])),
    //         60,
    //         ShapeStyle::from(&BLUE.mix(0.5)).stroke_width(1),
    //         &|coord, size, style| EmptyElement::at(coord) + Circle::new((0, 0), size, style),
    //     ))?;

    //     root.present()?;
    // }

    println!("png writing...");

    let last_state = state_list.back().unwrap().clone();

    let root = BitMapBackend::new("dpmm.png", (1600, 900)).into_drawing_area();

    root.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(&root)
        .margin(10)
        .set_all_label_area_size(50)
        .build_cartesian_2d(-30.0..30.0, -30.0..30.0)?;

    chart
        .configure_mesh()
        .x_labels(10)
        .y_labels(10)
        .disable_mesh()
        .x_label_formatter(&|v| format!("{:.1}", v))
        .y_label_formatter(&|v| format!("{:.1}", v))
        .draw()?;

    chart.draw_series(PointSeries::of_element(
        x.iter().map(|xi| (xi[0], xi[1])).into_iter(),
        2,
        ShapeStyle::from(&BLACK.mix(0.1)).filled(),
        &|coord, size, style| EmptyElement::at(coord) + Circle::new((0, 0), size, style),
    ))?;
    chart.draw_series(PointSeries::of_element(
        last_state
            .s_inv()
            .iter()
            .map(|(k, _)| last_state.theta().get(k).unwrap())
            .map(|theta_k| (theta_k.mu()[0], theta_k.mu()[1])),
        60,
        ShapeStyle::from(&BLUE.mix(0.5)).stroke_width(1),
        &|coord, size, style| EmptyElement::at(coord) + Circle::new((0, 0), size, style),
    ))?;

    root.present()?;

    Ok(())
}
