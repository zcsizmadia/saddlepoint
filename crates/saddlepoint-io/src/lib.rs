//! Free-format MPS reader.
//!
//! Supports NAME, OBJSENSE (inline or on the next line), ROWS, COLUMNS
//! (integer markers are read and ignored: the LP relaxation is returned),
//! RHS (an RHS on the objective row sets the constant to its negation),
//! RANGES and BOUNDS (UP, LO, FX, FR, MI, PL, BV, LI, UI). Names must not
//! contain spaces. Only the first RHS, RANGES and BOUNDS set is used.
//!
//! The existing `mps` crate was not used: it does not return the problem in
//! the bound form the solver needs and the conversion is most of the work.

use std::collections::HashMap;
use std::io::{BufRead, BufReader, Read};
use std::path::Path;

use saddlepoint_core::{CsrMatrix, LpProblem};
use thiserror::Error;

const INF: f64 = f64::INFINITY;

#[derive(Debug, Error)]
pub enum MpsError {
    #[error("i/o error: {0}")]
    Io(#[from] std::io::Error),
    #[error("line {line}: {msg}")]
    Parse { line: usize, msg: String },
}

#[derive(Clone, Copy, PartialEq)]
enum Section {
    None,
    Name,
    ObjSense,
    Rows,
    Columns,
    Rhs,
    Ranges,
    Bounds,
    Unsupported,
    End,
}

#[derive(Clone, Copy, PartialEq)]
enum RowKind {
    E,
    L,
    G,
}

pub fn read_mps_file(path: impl AsRef<Path>) -> Result<LpProblem, MpsError> {
    let path = path.as_ref();
    let mut p = read_mps(std::fs::File::open(path)?)?;
    if p.name.is_empty() {
        p.name = path.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
    }
    Ok(p)
}

pub fn read_mps(reader: impl Read) -> Result<LpProblem, MpsError> {
    let mut name = String::new();
    let mut maximize = false;
    let mut obj_row: Option<String> = None;
    let mut row_index: HashMap<String, usize> = HashMap::new();
    let mut row_names = Vec::new();
    let mut row_kind = Vec::new();
    let mut col_index: HashMap<String, usize> = HashMap::new();
    let mut col_names = Vec::new();
    let mut is_int = Vec::new();
    let mut c: Vec<f64> = Vec::new();
    let mut triplets = Vec::new();
    let mut rhs: Vec<f64> = Vec::new();
    let mut range: Vec<Option<f64>> = Vec::new();
    let mut c0 = 0.0;
    let mut lv: Vec<f64> = Vec::new();
    let mut uv: Vec<f64> = Vec::new();
    let mut upper_set: Vec<bool> = Vec::new();
    let mut ignored_rows: std::collections::HashSet<String> = Default::default();
    let (mut rhs_set, mut ranges_set, mut bounds_set): (Option<String>, Option<String>, Option<String>) =
        (None, None, None);
    let mut in_int = false;
    let mut section = Section::None;

    for (lineno, line) in BufReader::new(reader).lines().enumerate() {
        let line = line?;
        let lineno = lineno + 1;
        let err = |msg: String| MpsError::Parse { line: lineno, msg };
        if line.starts_with('*') || line.trim().is_empty() {
            continue;
        }
        let fields: Vec<&str> = line.split_whitespace().collect();
        if !line.starts_with([' ', '\t']) {
            section = match fields[0].to_ascii_uppercase().as_str() {
                "NAME" => {
                    name = fields.get(1).copied().unwrap_or("").to_string();
                    Section::Name
                }
                "OBJSENSE" => {
                    if let Some(s) = fields.get(1) {
                        maximize = parse_sense(s).ok_or_else(|| err(format!("bad OBJSENSE {s}")))?;
                    }
                    Section::ObjSense
                }
                "OBJSENCE" => Section::ObjSense,
                "ROWS" => Section::Rows,
                "COLUMNS" => Section::Columns,
                "RHS" => Section::Rhs,
                "RANGES" => Section::Ranges,
                "BOUNDS" => Section::Bounds,
                "ENDATA" => Section::End,
                "OBJSENSE_MAX" | "MAX" | "MAXIMIZE" | "MIN" | "MINIMIZE" if section == Section::ObjSense => {
                    maximize = parse_sense(fields[0]).unwrap_or(false);
                    Section::ObjSense
                }
                "QUADOBJ" | "QMATRIX" | "QSECTION" | "CSECTION" | "SOS" | "INDICATORS" => {
                    return Err(err(format!("section {} is not supported", fields[0])));
                }
                _ => Section::Unsupported,
            };
            if section == Section::Unsupported {
                return Err(err(format!("unknown section {}", fields[0])));
            }
            if section == Section::End {
                break;
            }
            continue;
        }

        match section {
            Section::ObjSense => {
                maximize = parse_sense(fields[0]).ok_or_else(|| err(format!("bad OBJSENSE {}", fields[0])))?;
            }
            Section::Rows => {
                if fields.len() < 2 {
                    return Err(err("ROWS line needs a type and a name".into()));
                }
                let kind = match fields[0].to_ascii_uppercase().as_str() {
                    "N" => {
                        if obj_row.is_none() {
                            obj_row = Some(fields[1].to_string());
                        } else {
                            ignored_rows.insert(fields[1].to_string());
                        }
                        continue;
                    }
                    "E" => RowKind::E,
                    "L" => RowKind::L,
                    "G" => RowKind::G,
                    t => return Err(err(format!("unknown row type {t}"))),
                };
                if row_index.insert(fields[1].to_string(), row_names.len()).is_some() {
                    return Err(err(format!("duplicate row {}", fields[1])));
                }
                row_names.push(fields[1].to_string());
                row_kind.push(kind);
                rhs.push(0.0);
                range.push(None);
            }
            Section::Columns => {
                if fields.len() >= 3 && fields[1].trim_matches('\'').eq_ignore_ascii_case("MARKER") {
                    match fields[2].trim_matches('\'').to_ascii_uppercase().as_str() {
                        "INTORG" => in_int = true,
                        "INTEND" => in_int = false,
                        m => return Err(err(format!("unknown marker {m}"))),
                    }
                    continue;
                }
                if fields.len() < 3 || fields.len().is_multiple_of(2) {
                    return Err(err("COLUMNS line needs a column and row/value pairs".into()));
                }
                let j = match col_index.get(fields[0]) {
                    Some(&j) => j,
                    None => {
                        let j = col_names.len();
                        col_index.insert(fields[0].to_string(), j);
                        col_names.push(fields[0].to_string());
                        is_int.push(in_int);
                        c.push(0.0);
                        lv.push(0.0);
                        uv.push(INF);
                        upper_set.push(false);
                        j
                    }
                };
                for pair in fields[1..].chunks(2) {
                    let v = parse_num(pair[1]).ok_or_else(|| err(format!("bad number {}", pair[1])))?;
                    if obj_row.as_deref() == Some(pair[0]) {
                        c[j] += v;
                    } else if let Some(&i) = row_index.get(pair[0]) {
                        triplets.push((i, j, v));
                    } else if !ignored_rows.contains(pair[0]) {
                        return Err(err(format!("unknown row {}", pair[0])));
                    }
                }
            }
            Section::Rhs | Section::Ranges => {
                // Optional set name: an odd field count means it is present.
                let (set, rest) = if fields.len() % 2 == 1 { (fields[0], &fields[1..]) } else { ("", &fields[..]) };
                let chosen = if section == Section::Rhs { &mut rhs_set } else { &mut ranges_set };
                if chosen.get_or_insert_with(|| set.to_string()) != set {
                    continue;
                }
                for pair in rest.chunks(2) {
                    let v = parse_num(pair[1]).ok_or_else(|| err(format!("bad number {}", pair[1])))?;
                    if obj_row.as_deref() == Some(pair[0]) {
                        if section == Section::Rhs {
                            c0 = -v;
                        }
                    } else if let Some(&i) = row_index.get(pair[0]) {
                        if section == Section::Rhs {
                            rhs[i] = v;
                        } else {
                            range[i] = Some(v);
                        }
                    } else if !ignored_rows.contains(pair[0]) {
                        return Err(err(format!("unknown row {}", pair[0])));
                    }
                }
            }
            Section::Bounds => {
                let kind = fields[0].to_ascii_uppercase();
                let needs_value = !matches!(kind.as_str(), "FR" | "MI" | "PL" | "BV");
                // Forms: TYPE SET COL [VAL] or TYPE COL [VAL].
                let (set, col, val) = match (fields.len(), needs_value) {
                    (4, true) | (3, false) => (fields[1], fields[2], fields.get(3).copied()),
                    (3, true) | (2, false) => ("", fields[1], fields.get(2).copied()),
                    (4, false) => (fields[1], fields[2], None),
                    _ => return Err(err("malformed BOUNDS line".into())),
                };
                if bounds_set.get_or_insert_with(|| set.to_string()) != set {
                    continue;
                }
                let j = *col_index.get(col).ok_or_else(|| err(format!("unknown column {col}")))?;
                let v = match val {
                    Some(s) => parse_num(s).ok_or_else(|| err(format!("bad number {s}")))?,
                    None => 0.0,
                };
                match kind.as_str() {
                    "UP" | "UI" => {
                        uv[j] = v;
                        upper_set[j] = true;
                        // Classic MPS rule: a negative upper bound with the
                        // default lower bound makes the lower bound -inf.
                        if v < 0.0 && lv[j] == 0.0 {
                            lv[j] = -INF;
                        }
                    }
                    "LO" | "LI" => lv[j] = v,
                    "FX" => {
                        lv[j] = v;
                        uv[j] = v;
                        upper_set[j] = true;
                    }
                    "FR" => {
                        lv[j] = -INF;
                        uv[j] = INF;
                    }
                    "MI" => lv[j] = -INF,
                    "PL" => uv[j] = INF,
                    "BV" => {
                        lv[j] = 0.0;
                        uv[j] = 1.0;
                        upper_set[j] = true;
                    }
                    t => return Err(err(format!("unsupported bound type {t}"))),
                }
            }
            Section::Name => {}
            Section::None => return Err(err("data before the first section".into())),
            Section::Unsupported | Section::End => unreachable!(),
        }
    }

    let m = row_names.len();
    let mut lc = vec![0.0; m];
    let mut uc = vec![0.0; m];
    for i in 0..m {
        let b = rhs[i];
        let (lo, hi) = match (row_kind[i], range[i]) {
            (RowKind::E, None) => (b, b),
            (RowKind::L, None) => (-INF, b),
            (RowKind::G, None) => (b, INF),
            (RowKind::E, Some(r)) if r >= 0.0 => (b, b + r),
            (RowKind::E, Some(r)) => (b + r, b),
            (RowKind::L, Some(r)) => (b - r.abs(), b),
            (RowKind::G, Some(r)) => (b, b + r.abs()),
        };
        lc[i] = lo;
        uc[i] = hi;
    }
    let n = col_names.len();
    let _ = is_int; // integrality is dropped: the LP relaxation is returned
    if maximize {
        c.iter_mut().for_each(|v| *v = -*v);
        c0 = -c0;
    }
    Ok(LpProblem {
        name,
        a: CsrMatrix::from_triplets(m, n, &triplets),
        c,
        c0,
        lc,
        uc,
        lv,
        uv,
        maximize,
        row_names,
        col_names,
    })
}

fn parse_sense(s: &str) -> Option<bool> {
    match s.to_ascii_uppercase().as_str() {
        "MAX" | "MAXIMIZE" | "MAXIMISE" => Some(true),
        "MIN" | "MINIMIZE" | "MINIMISE" => Some(false),
        _ => None,
    }
}

fn parse_num(s: &str) -> Option<f64> {
    let v: f64 = s.parse().ok()?;
    if v.is_nan() { None } else { Some(v) }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = "\
NAME          TESTLP
OBJSENSE
    MAX
ROWS
 N  COST
 L  LIM1
 G  LIM2
 E  MYEQN
 E  R4
COLUMNS
    X1        COST         1.0   LIM1         1.0
    X1        LIM2         1.0
    MARKER    'MARKER'     'INTORG'
    X2        COST         2.0   LIM1         1.0
    X2        MYEQN       -1.0
    MARKER    'MARKER'     'INTEND'
    X3        COST        -1.0   MYEQN        1.0
    X3        R4           1.0
RHS
    RHS       COST        -3.5
    RHS       LIM1         4.0   LIM2         1.0
    RHS       MYEQN        7.0   R4           2.0
RANGES
    RNG       LIM1         2.5   R4          -1.0
BOUNDS
 UP BND       X1           4.0
 MI BND       X2
 UP BND       X2           1.0
 FR BND       X3
ENDATA
";

    #[test]
    fn parses_all_sections() {
        let p = read_mps(SAMPLE.as_bytes()).unwrap();
        assert_eq!(p.name, "TESTLP");
        assert!(p.maximize);
        assert_eq!(p.c, vec![-1.0, -2.0, 1.0]);
        assert_eq!(p.c0, -3.5);
        assert_eq!(p.lc, vec![1.5, 1.0, 7.0, 1.0]);
        assert_eq!(p.uc, vec![4.0, INF, 7.0, 2.0]);
        assert_eq!(p.lv, vec![0.0, -INF, -INF]);
        assert_eq!(p.uv, vec![4.0, 1.0, INF]);
        assert_eq!(p.a.nnz(), 6);
        assert_eq!(p.a.row(2), (&[1usize, 2][..], &[-1.0, 1.0][..]));
        p.validate().unwrap();
    }

    #[test]
    fn rejects_unknown_rows() {
        let bad = "NAME X\nROWS\n N obj\nCOLUMNS\n    x  nope  1\nENDATA\n";
        assert!(matches!(read_mps(bad.as_bytes()), Err(MpsError::Parse { line: 5, .. })));
    }
}
