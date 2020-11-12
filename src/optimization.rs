use std::error::Error;
use std::ops::Add;

pub struct LogDistribution<'a> {
    params: Vec<&'a mut f64>,
    l: Box<dyn Fn() -> Result<(f64, Vec<(&'a f64, f64)>), Box<dyn Error>>>,
}

impl<'a> LogDistribution<'a> {
    pub fn new(
        params: Vec<&'a mut f64>,
        l: Box<dyn Fn() -> Result<(f64, Vec<(&'a f64, f64)>), Box<dyn Error>>>,
    ) -> Self {
        Self { params, l }
    }
}

pub struct LogDistributionSum<'a>(Vec<LogDistribution<'a>>);

impl<'a> Add for LogDistribution<'a> {
    type Output = LogDistributionSum<'a>;

    fn add(self, rhs: LogDistribution<'a>) -> Self::Output {
        LogDistributionSum(vec![self, rhs])
    }
}

impl<'a> Add<LogDistribution<'a>> for LogDistributionSum<'a> {
    type Output = LogDistributionSum<'a>;

    fn add(mut self, rhs: LogDistribution<'a>) -> Self::Output {
        self.0.push(rhs);

        self
    }
}
