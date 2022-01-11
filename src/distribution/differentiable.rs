use crate::DistributionError;
use crate::{DependentJoint, Distribution, IndependentJoint, RandomVariable};
use rand::prelude::*;
use std::{
    collections::HashMap,
    fmt::Debug,
    ops::{BitAnd, Mul},
};

pub trait DifferentiableDistribution: Distribution {
    fn log_diff(&self, x: &Self::Value, theta: &Self::Condition) -> Vec<f64>;
}
