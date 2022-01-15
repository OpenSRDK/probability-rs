use crate::DistributionError;
use crate::{DependentJoint, Distribution, IndependentJoint, RandomVariable};
use rand::prelude::*;
use std::{
    collections::HashMap,
    fmt::Debug,
    ops::{BitAnd, Mul},
};

pub mod condition;
pub mod dependent_joint;
pub mod value;

pub use condition::*;
pub use value::*;
