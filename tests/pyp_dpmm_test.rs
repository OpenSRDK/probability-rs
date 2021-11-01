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

    let np = vec![
        ExactEllipticalParams::new(vec![0.0, 20.0], DiagonalMatrix::identity(2).mat()).unwrap(),
        ExactEllipticalParams::new(vec![-20.0, 0.0], DiagonalMatrix::identity(2).mat()).unwrap(),
        ExactEllipticalParams::new(vec![20.0, -20.0], DiagonalMatrix::identity(2).mat()).unwrap(),
    ];
    let x = np
        .iter()
        .flat_map(|np| (0..20).into_iter().map(move |i| (np, i)))
        .map(|(np, _)| -> Result<Vec<f64>, DistributionError> {
            MultivariateNormal::new().sample(np, &mut rng)
        })
        .collect::<Result<Vec<_>, _>>()?;

    println!("x生成完了");

    let n = x.len();

    let alpha = 2.4;
    let d = 0.1;
    let pyp_params = PitmanYorProcessParams::new(alpha, d)?;

    let g0 = BaselineMeasure::new(InstantDistribution::new(
        &|_: &ExactEllipticalParams, _: &()| Ok(0.1),
        &|_, rng| {
            ExactEllipticalParams::new(
                vec![
                    30.0 * rng.gen_range(-1.0..=1.0),
                    30.0 * rng.gen_range(-1.0..=1.0),
                ],
                DiagonalMatrix::identity(2).mat(),
            )
        },
    ));

    let mh_proposal = InstantDistribution::new(
        &|x: &ExactEllipticalParams, theta: &ExactEllipticalParams| {
            Ok(MultivariateNormal::new().p(
                &x.mu(),
                &ExactEllipticalParams::new(
                    theta.mu().clone(),
                    (3.0 * DiagonalMatrix::identity(2)).mat(),
                )?,
            )?)
        },
        &|theta, rng| {
            let mu = MultivariateNormal::new().sample(
                &ExactEllipticalParams::new(
                    theta.mu().clone(),
                    (3.0 * DiagonalMatrix::identity(2)).mat(),
                )?,
                rng,
            )?;
            ExactEllipticalParams::new(mu, DiagonalMatrix::identity(2).mat())
        },
    );

    const ITER: usize = 150;
    const BURNIN: usize = 50;

    let mut s_list = LinkedList::<ClusterSwitch>::new();
    s_list.push_back(ClusterSwitch::new(
        (1u32..=n as u32).into_iter().collect::<Vec<_>>(),
    )?);

    let mut theta_list = LinkedList::<HashMap<_, _>>::new();
    theta_list.push_back(
        (1..=n as u32)
            .into_iter()
            .map(|k| {
                (
                    k,
                    ExactEllipticalParams::new(vec![0.0; 2], DiagonalMatrix::identity(2).mat())
                        .unwrap(),
                )
            })
            .collect::<HashMap<u32, ExactEllipticalParams>>(),
    );

    for iter in 0..ITER {
        println!("iteration {}", iter);

        let s = s_list.back().unwrap();
        let theta = theta_list.back().unwrap();

        let (new_s, new_theta) = rayon::join(
            || -> Result<_, DistributionError> {
                let new_s = {
                    let likelihood = MultivariateNormal::new().switch(
                        theta,
                        ExactEllipticalParams::new(
                            vec![0.0; 2],
                            DiagonalMatrix::identity(2).mat(),
                        )?,
                    );

                    let sampler = PitmanYorGibbsSampler::new(&pyp_params, s, &x, &likelihood);

                    sampler.sample(Some(&|i| [i as u8; 32]))?
                };

                Ok(new_s)
            },
            || -> Result<_, DistributionError> {
                let keys = theta.keys().into_iter().map(|&k| k).collect::<Vec<_>>();
                let new_theta = keys
                    .into_par_iter()
                    .map(|k| {
                        let x_in_k = PitmanYorProcessParams::x_in_cluster(&x, s.s(), k);

                        (k, x_in_k)
                    })
                    .filter(|(_, x_in_k)| x_in_k.len() != 0)
                    .map(|(k, x_in_k)| {
                        let x_likelihood = vec![MultivariateNormal::new(); x_in_k.len()]
                            .into_iter()
                            .joint();
                        let mh_sampler = MetropolisHastingsSampler::new(
                            &x_in_k,
                            &x_likelihood,
                            &g0.distr,
                            &mh_proposal,
                        );

                        let mut rng = thread_rng();

                        let theta_k = mh_sampler
                            .sample(4, theta.get(&k).unwrap().clone(), &mut rng)
                            .unwrap();

                        (k, theta_k)
                    })
                    .collect::<HashMap<_, _>>();

                Ok(new_theta)
            },
        );

        if iter <= BURNIN {
            s_list.clear();
            theta_list.clear();
        }
        s_list.push_back(new_s?);
        theta_list.push_back(new_theta?);
    }

    let mut accumulated_s = Vec::<SamplesDistribution<_>>::new();
    let mut e_theta = HashMap::<u32, (usize, ExactEllipticalParams)>::new();

    for (s_t, theta_t) in s_list.into_iter().zip(theta_list.into_iter()) {
        accumulated_s
            .par_iter_mut()
            .zip(s_t.s().par_iter())
            .for_each(|(asi, &sti)| asi.push(sti));

        for (k, theta_tk) in theta_t {
            let entry = e_theta.entry(k).or_insert((
                0,
                ExactEllipticalParams::new(vec![0.0, 0.0], DiagonalMatrix::identity(2).mat())?,
            ));
            let (new_mu, new_lsigma) = theta_tk.eject();
            let new_weight = (1.0 / (entry.0 + 1) as f64).max(0.05);

            entry.1 = ExactEllipticalParams::new(
                ((1.0 - new_weight) * entry.1.mu().to_vec().col_mat()
                    + new_weight * new_mu.col_mat())
                .vec(),
                new_lsigma,
            )
            .unwrap();
        }

        let accumulated_clusters = ClusterSwitch::new(
            accumulated_s
                .iter()
                .map(|asi| -> Result<_, DistributionError> { Ok(asi.mode()?.clone()) })
                .collect::<Result<_, _>>()?,
        )?;

        let root = BitMapBackend::gif("dpmm.gif", (1600, 900), 0_500)?.into_drawing_area();

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
            accumulated_clusters
                .s_inv()
                .iter()
                .map(|(k, _)| e_theta.get(k).unwrap())
                .map(|(_, theta_k)| (theta_k.mu()[0], theta_k.mu()[1])),
            60,
            ShapeStyle::from(&BLUE.mix(0.5)).stroke_width(1),
            &|coord, size, style| EmptyElement::at(coord) + Circle::new((0, 0), size, style),
        ))?;

        root.present()?;
    }

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
        e_theta
            .iter()
            .map(|(_, (_, theta_k))| (theta_k.mu()[0], theta_k.mu()[1])),
        60,
        ShapeStyle::from(&BLUE.mix(0.5)).stroke_width(1),
        &|coord, size, style| EmptyElement::at(coord) + Circle::new((0, 0), size, style),
    ))?;
    root.present()?;

    Ok(())
}
