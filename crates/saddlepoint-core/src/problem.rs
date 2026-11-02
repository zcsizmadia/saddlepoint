//! LP problem representation.
//!
//! ```text
//! minimize    cᵀx + c0
//! subject to  lc ≤ A x ≤ uc
//!             lv ≤ x   ≤ uv
//! ```
//!
//! Infinite bounds are `f64::INFINITY` / `f64::NEG_INFINITY`. Equality rows
//! have `lc == uc`.

use thiserror::Error;

/// Compressed sparse row matrix.
#[derive(Clone, Debug, PartialEq)]
pub struct CsrMatrix {
    pub nrows: usize,
    pub ncols: usize,
    /// Length `nrows + 1`.
    pub row_ptr: Vec<usize>,
    /// Column index of each nonzero, sorted within a row.
    pub col_idx: Vec<usize>,
    pub values: Vec<f64>,
}

impl CsrMatrix {
    pub fn zeros(nrows: usize, ncols: usize) -> Self {
        Self { nrows, ncols, row_ptr: vec![0; nrows + 1], col_idx: Vec::new(), values: Vec::new() }
    }

    /// Builds a CSR matrix from `(row, col, value)` triplets. Duplicates are
    /// summed and explicit zeros dropped.
    pub fn from_triplets(nrows: usize, ncols: usize, triplets: &[(usize, usize, f64)]) -> Self {
        let mut counts = vec![0usize; nrows + 1];
        for &(r, c, _) in triplets {
            assert!(r < nrows && c < ncols, "triplet ({r}, {c}) out of range");
            counts[r + 1] += 1;
        }
        for i in 0..nrows {
            counts[i + 1] += counts[i];
        }
        let mut next = counts.clone();
        let mut cols = vec![0usize; triplets.len()];
        let mut vals = vec![0.0; triplets.len()];
        for &(r, c, v) in triplets {
            let k = next[r];
            cols[k] = c;
            vals[k] = v;
            next[r] += 1;
        }
        // Sort each row and merge duplicates.
        let mut row_ptr = Vec::with_capacity(nrows + 1);
        let mut col_idx = Vec::with_capacity(triplets.len());
        let mut values = Vec::with_capacity(triplets.len());
        row_ptr.push(0);
        let mut scratch: Vec<(usize, f64)> = Vec::new();
        for i in 0..nrows {
            scratch.clear();
            scratch.extend((counts[i]..counts[i + 1]).map(|k| (cols[k], vals[k])));
            scratch.sort_unstable_by_key(|e| e.0);
            let mut k = 0;
            while k < scratch.len() {
                let (c, mut v) = scratch[k];
                k += 1;
                while k < scratch.len() && scratch[k].0 == c {
                    v += scratch[k].1;
                    k += 1;
                }
                if v != 0.0 {
                    col_idx.push(c);
                    values.push(v);
                }
            }
            row_ptr.push(col_idx.len());
        }
        Self { nrows, ncols, row_ptr, col_idx, values }
    }

    pub fn nnz(&self) -> usize {
        self.values.len()
    }

    pub fn row(&self, i: usize) -> (&[usize], &[f64]) {
        let r = self.row_ptr[i]..self.row_ptr[i + 1];
        (&self.col_idx[r.clone()], &self.values[r])
    }

    pub fn transpose(&self) -> Self {
        let mut counts = vec![0usize; self.ncols + 1];
        for &c in &self.col_idx {
            counts[c + 1] += 1;
        }
        for j in 0..self.ncols {
            counts[j + 1] += counts[j];
        }
        let mut next = counts.clone();
        let mut col_idx = vec![0usize; self.nnz()];
        let mut values = vec![0.0; self.nnz()];
        for i in 0..self.nrows {
            let (cols, vals) = self.row(i);
            for (&c, &v) in cols.iter().zip(vals) {
                let k = next[c];
                col_idx[k] = i;
                values[k] = v;
                next[c] += 1;
            }
        }
        Self { nrows: self.ncols, ncols: self.nrows, row_ptr: counts, col_idx, values }
    }

    /// `out = A x` (serial reference implementation).
    pub fn mul_vec(&self, x: &[f64], out: &mut [f64]) {
        for (i, o) in out.iter_mut().enumerate() {
            let (cols, vals) = self.row(i);
            *o = cols.iter().zip(vals).map(|(&c, &v)| v * x[c]).sum();
        }
    }

    /// Scales entry `(i, j)` by `row[i] * col[j]`.
    pub fn scale(&mut self, row: &[f64], col: &[f64]) {
        for (i, &ri) in row.iter().enumerate().take(self.nrows) {
            for k in self.row_ptr[i]..self.row_ptr[i + 1] {
                self.values[k] *= ri * col[self.col_idx[k]];
            }
        }
    }
}

/// A linear program in the form documented at module level.
#[derive(Clone, Debug)]
pub struct LpProblem {
    pub name: String,
    pub a: CsrMatrix,
    pub c: Vec<f64>,
    pub c0: f64,
    pub lc: Vec<f64>,
    pub uc: Vec<f64>,
    pub lv: Vec<f64>,
    pub uv: Vec<f64>,
    /// `true` if the original model maximized. `c` and `c0` are already
    /// negated so the stored problem always minimizes.
    pub maximize: bool,
    pub row_names: Vec<String>,
    pub col_names: Vec<String>,
}

#[derive(Debug, Error, PartialEq)]
pub enum ProblemError {
    #[error("{what} has length {got}, expected {expected}")]
    Length { what: &'static str, got: usize, expected: usize },
    #[error("{what} {index} has lower bound {lo} above upper bound {hi}")]
    CrossedBounds { what: &'static str, index: usize, lo: f64, hi: f64 },
    #[error("{what} {index} is NaN")]
    NaN { what: &'static str, index: usize },
}

impl LpProblem {
    pub fn nrows(&self) -> usize {
        self.a.nrows
    }

    pub fn ncols(&self) -> usize {
        self.a.ncols
    }

    pub fn validate(&self) -> Result<(), ProblemError> {
        let (m, n) = (self.nrows(), self.ncols());
        for (what, v, len) in
            [("c", &self.c, n), ("lv", &self.lv, n), ("uv", &self.uv, n), ("lc", &self.lc, m), ("uc", &self.uc, m)]
        {
            if v.len() != len {
                return Err(ProblemError::Length { what, got: v.len(), expected: len });
            }
        }
        if self.a.row_ptr.len() != m + 1 || self.a.col_idx.len() != self.a.values.len() {
            return Err(ProblemError::Length { what: "A", got: self.a.row_ptr.len(), expected: m + 1 });
        }
        for (what, lo, hi) in [("variable", &self.lv, &self.uv), ("constraint", &self.lc, &self.uc)] {
            for (index, (&l, &u)) in lo.iter().zip(hi).enumerate() {
                if l.is_nan() || u.is_nan() {
                    return Err(ProblemError::NaN { what, index });
                }
                if l > u {
                    return Err(ProblemError::CrossedBounds { what, index, lo: l, hi: u });
                }
            }
        }
        for (index, v) in self.c.iter().chain(&self.a.values).enumerate() {
            if v.is_nan() {
                return Err(ProblemError::NaN { what: "coefficient", index });
            }
        }
        Ok(())
    }

    /// Objective value of `x` in the original sense (max problems report the
    /// maximized value).
    pub fn objective(&self, x: &[f64]) -> f64 {
        let v = self.c.iter().zip(x).map(|(c, x)| c * x).sum::<f64>() + self.c0;
        if self.maximize { -v } else { v }
    }
}
