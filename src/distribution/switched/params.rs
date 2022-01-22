use std::fmt::Debug;

use crate::{Distribution, RandomVariable, SwitchedDistribution};

#[derive(Clone, Debug)]
pub enum SwitchedParams<U>
where
    U: Clone + Debug + Send + Sync,
{
    Key(u32),
    Direct(U),
}
