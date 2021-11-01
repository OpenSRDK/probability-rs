use std::collections::{HashMap, HashSet};

use crate::DistributionError;

#[derive(Clone, Debug)]
pub struct ClusterSwitch {
    s: Vec<u32>,
    s_inv: HashMap<u32, HashSet<usize>>,
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
        let mut s_inv = HashMap::new();
        let mut max_k = 0;

        for (i, &si) in s.iter().enumerate() {
            if si == 0 {
                return Err(DistributionError::InvalidParameters(
                    ClusterSwitchError::SMustBePositive.into(),
                ));
            }
            s_inv.entry(si).or_insert(HashSet::new()).insert(i);

            if max_k < si {
                max_k = si
            }
        }

        Ok(Self { s, s_inv, max_k })
    }

    pub fn s(&self) -> &Vec<u32> {
        &self.s
    }

    pub fn s_inv(&self) -> &HashMap<u32, HashSet<usize>> {
        &self.s_inv
    }

    pub fn set_s(&mut self, i: usize, si: u32) -> u32 {
        self.s_inv
            .entry(self.s[i])
            .or_insert(HashSet::new())
            .remove(&i);
        if self.s_inv.get(&self.s[i]).unwrap().len() == 0 {
            self.s_inv.remove(&self.s[i]);
        }

        if si == 0 {
            self.max_k += 1;
            self.s[i] = self.max_k;
            self.s_inv
                .entry(self.s[i])
                .or_insert(HashSet::new())
                .insert(i);

            return self.max_k;
        }

        self.s[i] = si;
        self.s_inv
            .entry(self.s[i])
            .or_insert(HashSet::new())
            .insert(i);

        if self.max_k < si {
            self.max_k = si;
        }

        si
    }

    pub fn n(&self, k: u32) -> usize {
        match self.s_inv.get(&k) {
            Some(v) => v.len(),
            None => 0,
        }
    }

    pub fn clusters_len(&self) -> usize {
        self.s_inv.len()
    }
}
