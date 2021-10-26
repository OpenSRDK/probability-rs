use crate::{Distribution, RandomVariable};

#[derive(Clone, Debug)]
pub struct Conditioned<'a, T, U, D>
where
    T: RandomVariable,
    U: RandomVariable,
    D: Distribution<T = T, U = U>,
{
    original: &'a D,
    condition: &'a U,
}

impl<'a, T, U, D> Conditioned<'a, T, U, D>
where
    T: RandomVariable,
    U: RandomVariable,
    D: Distribution<T = T, U = U>,
{
    pub fn new(original: &'a D, condition: &'a U) -> Self {
        Self {
            original,
            condition,
        }
    }
}

impl<'a, T, U, D> Distribution for Conditioned<'a, T, U, D>
where
    T: RandomVariable,
    U: RandomVariable,
    D: Distribution<T = T, U = U>,
{
    type T = T;
    type U = ();

    fn p(&self, x: &Self::T, _: &Self::U) -> Result<f64, crate::DistributionError> {
        self.original.p(x, self.condition)
    }

    fn sample(
        &self,
        _: &Self::U,
        rng: &mut rand::prelude::StdRng,
    ) -> Result<Self::T, crate::DistributionError> {
        self.original.sample(self.condition, rng)
    }
}
