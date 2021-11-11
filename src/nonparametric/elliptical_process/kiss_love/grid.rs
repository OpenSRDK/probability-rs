use crate::DistributionError;
use crate::{nonparametric::kernel_matrix, opensrdk_linear_algebra::*};
use opensrdk_kernel_method::KernelError;
use opensrdk_kernel_method::{Convolutable, Kernel};
use rayon::prelude::*;

#[derive(thiserror::Error, Debug)]
pub enum InducingGridError {
    #[error("empty")]
    Empty,
    #[error("dimension mismatch")]
    DimensionMismatch,
    #[error("NaN contamination")]
    NaNContamination,
    #[error("invalid range")]
    InvalidRange,
    #[error("points must be more than or equal to 2")]
    TooLessPoints,
}

#[derive(Clone, Debug)]
pub struct Axis {
    min: f64,
    max: f64,
    points: usize,
}

impl Axis {
    pub fn new(min: f64, max: f64, points: usize) -> Result<Self, DistributionError> {
        if max <= min {
            return Err(DistributionError::InvalidParameters(
                InducingGridError::InvalidRange.into(),
            ));
        }
        if points < 2 {
            return Err(DistributionError::InvalidParameters(
                InducingGridError::TooLessPoints.into(),
            ));
        }

        Ok(Self { min, max, points })
    }

    pub fn points(&self) -> usize {
        self.points
    }

    pub fn value(&self, index: usize) -> f64 {
        self.min + index as f64 * (self.max - self.min) / (self.points - 1) as f64
    }

    pub fn index(&self, value: f64) -> usize {
        ((value - self.min) / (self.max - self.min) * (self.points - 1) as f64) as usize
    }
}

#[derive(Clone, Debug)]
pub struct Grid {
    axes: Vec<Axis>,
}

impl Grid {
    pub fn new(axes: Vec<Axis>) -> Self {
        Self { axes }
    }

    pub fn from<T>(x: &[T], points: &[usize]) -> Result<Grid, DistributionError>
    where
        T: Convolutable,
    {
        let n = x.len();
        if n == 0 {
            return Err(DistributionError::InvalidParameters(
                InducingGridError::Empty.into(),
            ));
        }

        let d = x[0].data_len();
        if d == 0 {
            return Err(DistributionError::InvalidParameters(
                InducingGridError::Empty.into(),
            ));
        }

        let axis_factory = |(nd, &points_)| {
            let (min, max) = (0..n)
                .into_iter()
                .flat_map(|ni| (0..x[ni].parts_len()).map(move |pi| x[ni].part(pi)[nd]))
                .fold((0.0 / 0.0, 0.0 / 0.0), |sum, xnid: f64| {
                    (xnid.min(sum.0), xnid.max(sum.1))
                });

            Axis::new(min, max, points_)
        };

        let axes = (0..d)
            .into_iter()
            .zip(points)
            .map(axis_factory)
            .collect::<Result<Vec<_>, DistributionError>>()?;

        Ok(Grid::new(axes))
    }

    pub fn add(&mut self, axis: Axis) {
        self.axes.push(axis);
    }

    pub fn kuu(
        &self,
        kernel: &impl Kernel<Vec<f64>>,
        params: &[f64],
    ) -> Result<KroneckerMatrices, KernelError> {
        let d = self.axes.len();

        let k = self
            .axes
            .iter()
            .enumerate()
            .map(|(di, udi)| {
                let udi_array = (0..udi.points)
                    .into_iter()
                    .map(|pi| {
                        let mut value = vec![0.0; d];
                        value[di] = udi.value(pi);

                        value
                    })
                    .collect::<Vec<_>>();

                kernel_matrix(kernel, params, &udi_array, &udi_array.as_ref())
            })
            .collect::<Result<Vec<_>, KernelError>>()?;

        let ks = KroneckerMatrices::new(k);

        Ok(ks)
    }

    fn sparse_kronecker_prod(k: &[SparseMatrix]) -> SparseMatrix {
        let rows = k.par_iter().map(|kp| kp.rows).product::<usize>();
        let cols = k.par_iter().map(|kp| kp.cols).product::<usize>();
        let mut new_matrix = k[0].clone();
        let k_len = k.len();

        for p in (1..k_len).rev() {
            let lhs = new_matrix;
            let rhs = &k[p];
            new_matrix = SparseMatrix::new(rows, cols);

            for (&(li, lj), &lv) in lhs.elems.iter() {
                let istart = li * rhs.rows;
                let jstart = lj * rhs.cols;
                for (&(ri, rj), &rv) in rhs.elems.iter() {
                    let i = istart + ri;
                    let j = jstart + rj;

                    new_matrix[(i, j)] = lv * rv;
                }
            }
        }

        new_matrix
    }

    pub fn interpolation_weight<T>(&self, x: &[T]) -> Result<Vec<SparseMatrix>, DistributionError>
    where
        T: Convolutable,
    {
        let m = self.axes().par_iter().map(|ud| ud.points).product();
        let n = x.len();
        if n == 0 {
            return Err(DistributionError::InvalidParameters(
                InducingGridError::Empty.into(),
            ));
        }

        let p = x[0].parts_len();
        let d = x[0].data_len();
        if p == 0 || d == 0 {
            return Err(DistributionError::InvalidParameters(
                InducingGridError::Empty.into(),
            ));
        }

        if d != self.axes.len() {
            return Err(DistributionError::InvalidParameters(
                InducingGridError::DimensionMismatch.into(),
            ));
        }

        let wxpinidi_factory = |pi: usize, ni: usize| {
            move |di: usize| {
                let xpinidi = x[ni].part(pi)[di];
                let udi = &self.axes[di];

                let mut index = {
                    if xpinidi <= udi.min {
                        0
                    } else if udi.max <= xpinidi {
                        udi.points - 1
                    } else {
                        udi.index(xpinidi)
                    }
                };
                if index == udi.points - 1 {
                    index = udi.points - 2;
                }

                let udi1 = udi.value(index);
                let udi2 = udi.value(index + 1);
                // w * u1 + (1 - w) * u2 = x
                // w = (u2 - x) / (u2 - u1)
                let weight = (udi2 - xpinidi) as f64 / (udi2 - udi1) as f64;

                let mut sparse = SparseMatrix::new(udi.points, 1);
                sparse[(index, 0)] = weight;
                sparse[(index + 1, 0)] = 1.0 - weight;

                sparse
            }
        };

        let wxpini_factory = |pi| {
            move |ni| {
                let wxpinidi = (0..d)
                    .into_par_iter()
                    .map(wxpinidi_factory(pi, ni))
                    .collect::<Vec<_>>();
                let wxpini = Self::sparse_kronecker_prod(&wxpinidi);

                SparseMatrix::from(
                    m,
                    n,
                    wxpini
                        .elems
                        .iter()
                        .map(|(&(index, _), &value)| ((index, ni), value))
                        .collect(),
                )
            }
        };

        let wxpi_factory = |pi| {
            (0..n).into_par_iter().map(wxpini_factory(pi)).reduce(
                || SparseMatrix::new(m, n),
                |mut acc: SparseMatrix, v: SparseMatrix| {
                    acc.elems.extend(v.elems);
                    acc
                },
            )
        };

        let wx = (0..p).into_par_iter().map(wxpi_factory).collect::<Vec<_>>();

        Ok(wx)
    }

    pub fn axes(&self) -> &[Axis] {
        &self.axes
    }
}

#[cfg(test)]
mod tests {
    use super::{Axis, Grid};
    use crate::opensrdk_linear_algebra::*;

    #[test]
    fn it_works() {
        let grid = Grid::new(vec![Axis::new(0.0, 1.0, 2).unwrap(); 3]);

        // Each element of the vector x is a binary 0 or 1.
        // The elements of the sparse matrix in wx are arranged in the order in which they would be if the tree were made up of the first through nth elements of x.
        // If x is composed of only the largest or smallest values of Axes, then one of the elements of the sparse matrix in wx will be 1, and the order of the elements that are 1 is calculated from the tree.

        let x = vec![0.0, 1.0, 1.0];

        let x1 = &x[0] * (2f64.powi((x.len() as i32) - 1));
        let x2 = &x[1] * (2f64.powi((x.len() as i32) - 2));
        let x3 = &x[2] * (2f64.powi((x.len() as i32) - 3));
        let sum_x = (x1 + x2 + x3) as usize;

        let wx = grid.interpolation_weight(&[x]).unwrap();

        assert_eq!(wx[0][(sum_x, 0)], 1.0)
    }

    #[test]
    fn sparse() {
        let mut a = SparseMatrix::new(2, 2);
        a[(0, 0)] = 1.0;
        a[(0, 1)] = 2.0;
        a[(1, 0)] = 3.0;
        a[(1, 1)] = 4.0;

        let mut b = SparseMatrix::new(2, 2);
        b[(0, 0)] = 1.0;
        b[(0, 1)] = 2.0;
        b[(1, 0)] = 3.0;
        b[(1, 1)] = 4.0;

        let c = Grid::sparse_kronecker_prod(&[a.clone(), b.clone()]);

        for i in 0..a.rows {
            for j in 0..a.cols {
                for k in 0..b.rows {
                    for l in 0..b.cols {
                        let v1 = a[(i, j)] * b[(k, l)];
                        let v2 = c[(2 * i + k, 2 * j + l)];
                        assert_eq!(v1, v2)
                    }
                }
            }
        }
    }
}
