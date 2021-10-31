use std::collections::HashMap;

use crate::DistributionError;

#[derive(Clone, Debug)]
pub struct ClusterSwitch {
    s: Vec<u32>,
    n_map: HashMap<u32, usize>,
    max_k: u32,
}

#[derive(thiserror::Error, Debug)]
pub enum ClusterSwitchError {
    #[error("All elements of `s` must be positive")]
    SMustBePositive,
    #[error("Unknown error")]
    Unknown,
}

impl ClusterSwitch {
    pub fn new(s: Vec<u32>) -> Result<Self, DistributionError> {
        let mut n_map = HashMap::new();
        let mut max_k = 0;

        for &si in s.iter() {
            if si == 0 {
                return Err(DistributionError::InvalidParameters(
                    ClusterSwitchError::SMustBePositive.into(),
                ));
            }
            *n_map.entry(si).or_insert(0) += 1usize;

            if max_k < si {
                max_k = si
            }
        }

        Ok(Self { s, n_map, max_k })
    }

    pub fn s(&self) -> &Vec<u32> {
        &self.s
    }

    pub fn n_map(&self) -> &HashMap<u32, usize> {
        &self.n_map
    }

    pub fn set_s(&mut self, i: usize, si: u32) -> u32 {
        *self.n_map.entry(self.s[i]).or_insert(1) -= 1;
        if *self.n_map.get(&self.s[i]).unwrap() == 0 {
            self.n_map.remove(&self.s[i]);
        }

        if si == 0 {
            self.max_k += 1;
            self.s[i] = self.max_k;
            *self.n_map.entry(self.s[i]).or_insert(0) += 1;

            return self.max_k;
        }

        self.s[i] = si;
        *self.n_map.entry(self.s[i]).or_insert(0) += 1;
        if self.max_k < si {
            self.max_k = si;
        }

        si
    }

    pub fn n(&self, k: u32) -> usize {
        match self.n_map.get(&k) {
            Some(&v) => v,
            None => 0,
        }
    }

    pub fn clusters_len(&self) -> usize {
        self.n_map.len()
    }
}
