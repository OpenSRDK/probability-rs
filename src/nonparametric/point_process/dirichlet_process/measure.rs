use crate::nonparametric::{DiscreteMeasurableSpace, DiscreteMeasure};
use crate::RandomVariable;

#[derive(Clone, Debug)]
pub struct DirichletRandomMeasure<T>
where
    T: RandomVariable,
{
    w_theta: Vec<(f64, T)>,
}

impl<T> DiscreteMeasure for DirichletRandomMeasure<T>
where
    T: RandomVariable,
{
    fn measure(&self, a: DiscreteMeasurableSpace) -> f64 {
        a.iter().map(|&i| self.w_theta[i].0).sum::<f64>()
    }
}

impl<T> DirichletRandomMeasure<T>
where
    T: RandomVariable,
{
    pub fn new(w_theta: Vec<(f64, T)>) -> Self {
        Self { w_theta }
    }

    pub fn w_theta(&self) -> &Vec<(f64, T)> {
        &self.w_theta
    }
}
