extern crate blas_src;
extern crate lapack_src;
extern crate opensrdk_kernel_method;
extern crate opensrdk_linear_algebra;
extern crate opensrdk_probability;
extern crate plotters;
extern crate rayon;

use crate::opensrdk_probability::*;
use opensrdk_kernel_method::*;
use opensrdk_probability::nonparametric::*;
use plotters::{coord::Shift, prelude::*};
use rand::prelude::*;
use rand_distr::StandardNormal;

#[test]
fn test_main() {
    let is_not_ci = false;
    let is_gif = true;
    let exact = true;

    if is_not_ci {
        if is_gif {
            draw_gif(exact).unwrap();
        } else {
            draw_png(exact).unwrap();
        }
    }
}

fn func(x: f64) -> f64 {
    0.1 * x + x.sin() + 2.0 * (-x.powi(2)).exp()
}

fn samples(size: usize) -> Vec<(f64, f64)> {
    let mut rng = StdRng::from_seed([1; 32]);
    let mut rng2 = StdRng::from_seed([32; 32]);

    (0..size)
        .into_iter()
        .map(|_| {
            let x = rng2.gen_range(-8.0..=8.0);
            let y = func(x) + rng.sample::<f64, _>(StandardNormal);

            (x, y)
        })
        .collect()
}

fn draw(
    exact: bool, //Trueが入っていた
    size: usize, //2^(3+k)
    root: &DrawingArea<BitMapBackend, Shift>,
) -> Result<(), Box<dyn std::error::Error>> {
    let samples = samples(size);
    let x = samples.par_iter().map(|v| vec![v.0]).collect::<Vec<_>>();
    let y = samples.par_iter().map(|v| v.1).collect::<Vec<_>>();
    let kernel = RBF + Periodic;
    let theta = vec![1.0; kernel.params_len()]; //hyper parameterがtheta
    let sigma = 1.0;
    let base = BaseEllipticalProcessParams::new(kernel, x, theta, sigma)?;

    let x_axis = (-8.0..8.0).step(0.1);

    root.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(&root)
        .margin(10)
        .set_all_label_area_size(50)
        .build_cartesian_2d(-8.0..8.0, -6.0..6.0)?;

    chart
        .configure_mesh()
        .x_labels(10)
        .y_labels(10)
        .disable_mesh()
        .x_label_formatter(&|v| format!("{:.1}", v))
        .y_label_formatter(&|v| format!("{:.1}", v))
        .draw()?;

    chart.draw_series(LineSeries::new(
        x_axis.values().map(|x| (x, func(x))),
        &GREEN,
    ))?;

    chart.draw_series(PointSeries::of_element(
        samples.iter(),
        2,
        ShapeStyle::from(&BLACK.mix(0.1)).filled(),
        &|&coord, size, style| EmptyElement::at(coord) + Circle::new((0, 0), size, style),
    ))?;

    let result = if exact {
        let params = base.exact(&y)?;
        x_axis
            .values()
            .map(|xs: f64| {
                let np = params.gp_predict(&vec![xs]).unwrap();
                let mu = np.mu();
                let sigma = np.sigma();

                (xs, mu, sigma)
            })
            .collect::<Vec<_>>()
    } else {
        let params = base.sparse(
            &y,
            (0..=20)
                .into_iter()
                .map(|v| vec![v as f64 * 16.0 / 20.0 - 8.0])
                .collect::<Vec<_>>(),
        )?;
        x_axis
            .values()
            .map(|xs: f64| {
                let np = params.cp_predict(&vec![xs]).unwrap();
                let mu = np.mu();
                let sigma = np.sigma();

                (xs, mu, sigma)
            })
            .collect::<Vec<_>>()
    };

    // chart.draw_series(AreaSeries::new(
    //     x_axis.values().map(|xs| {
    //         let (mu, sigma) = exact_gp(RBF, &y, &x, &lkxx, vec![xs], &theta);

    //         (xs, mu + sigma)
    //     }),
    //     0.0,
    //     &BLUE.mix(0.5),
    // ))?;
    chart.draw_series(LineSeries::new(
        result.iter().map(|&(xs, mu, sigma)| (xs, mu + 3.0 * sigma)),
        &RED.mix(0.5),
    ))?;
    chart.draw_series(LineSeries::new(
        result.iter().map(|&(xs, mu, _)| (xs, mu)),
        &RGBColor(255, 0, 255).mix(0.5),
    ))?;
    chart.draw_series(LineSeries::new(
        result.iter().map(|&(xs, mu, sigma)| (xs, mu - 3.0 * sigma)),
        &BLUE.mix(0.5),
    ))?;

    root.present()?;

    Ok(())
}

fn draw_png(exact: bool) -> Result<(), Box<dyn std::error::Error>> {
    let root = BitMapBackend::new(
        if exact {
            "exact_gp.png"
        } else {
            "approx_gp.png"
        },
        (1600, 900),
    )
    .into_drawing_area();

    draw(exact, 1024, &root)?;

    Ok(())
}

fn draw_gif(exact: bool) -> Result<(), Box<dyn std::error::Error>> {
    let root = BitMapBackend::gif(
        if exact {
            "exact_gp.gif"
        } else {
            "approx_gp.gif"
        },
        (1600, 900),
        1_000,
    )?
    .into_drawing_area();

    for k in 0..8 {
        println!("iter: {}", k);
        draw(exact, 2usize.pow(3 + k), &root)?;
    }

    Ok(())
}
