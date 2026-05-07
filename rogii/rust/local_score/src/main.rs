use blake2::Blake2bVar;
use blake2::digest::{Update, VariableOutput};
use rayon::ThreadPoolBuilder;
use rayon::prelude::*;
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::env;
use std::fs;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

const DEFAULT_TRAIN_DIR: &str = "rogii/data/competition/train";
const DEFAULT_TRAIN_DIR_FROM_ROGII: &str = "data/competition/train";
const DEFAULT_FALLBACK_TVT: f64 = 11354.51;

type AnyResult<T> = Result<T, Box<dyn std::error::Error + Send + Sync>>;

#[derive(Debug, Clone)]
struct Opts {
    cmd: String,
    train_dir: PathBuf,
    seed: u64,
    n_folds: usize,
    fold: usize,
    limit: usize,
    wells: Option<Vec<String>>,
    strategy: String,
    fallback_tvt: f64,
    slope_cap: f64,
    slope_tail: usize,
    slope_min_points: usize,
    tail: usize,
    z_alpha: f64,
    z_slope_cap: f64,
    predictions_csv: Option<PathBuf>,
    truth_csv: Option<PathBuf>,
    export_dir: Option<PathBuf>,
    json: bool,
    top: usize,
    threads: usize,
}

#[derive(Debug)]
struct WellData {
    name: String,
    md: Vec<f64>,
    z: Vec<f64>,
    tvt: Vec<f64>,
    tvt_input: Vec<f64>,
}

#[derive(Debug)]
struct TruthRow {
    id: String,
    well: String,
    tvt: f64,
}

#[derive(Debug, Clone)]
struct WellMetric {
    well: String,
    rows: usize,
    rmse: f64,
    mae: f64,
    bias: f64,
}

#[derive(Debug)]
struct Metrics {
    rows: usize,
    wells: usize,
    rmse: f64,
    mae: f64,
    bias: f64,
    median_ae: f64,
    p90_ae: f64,
    mean_well_rmse: f64,
    elapsed_s: Option<f64>,
    rows_per_s: Option<f64>,
    per_well: Vec<WellMetric>,
}

struct Accum {
    rows: usize,
    sum_sq: f64,
    sum_abs: f64,
    sum_err: f64,
    abs_errs: Vec<f64>,
    per_well: Vec<WellMetric>,
}

impl Accum {
    fn new() -> Self {
        Self {
            rows: 0,
            sum_sq: 0.0,
            sum_abs: 0.0,
            sum_err: 0.0,
            abs_errs: Vec::new(),
            per_well: Vec::new(),
        }
    }

    fn add_well(&mut self, well: String, errs: &[f64]) {
        if errs.is_empty() {
            return;
        }
        let rows = errs.len();
        let mut sum_sq = 0.0;
        let mut sum_abs = 0.0;
        let mut sum_err = 0.0;
        for &e in errs {
            let ae = e.abs();
            sum_sq += e * e;
            sum_abs += ae;
            sum_err += e;
            self.abs_errs.push(ae);
        }
        self.rows += rows;
        self.sum_sq += sum_sq;
        self.sum_abs += sum_abs;
        self.sum_err += sum_err;
        self.per_well.push(WellMetric {
            well,
            rows,
            rmse: (sum_sq / rows as f64).sqrt(),
            mae: sum_abs / rows as f64,
            bias: sum_err / rows as f64,
        });
    }

    fn finish(mut self, elapsed_s: Option<f64>) -> Metrics {
        let rows = self.rows;
        let median_ae = percentile_in_place(&mut self.abs_errs, 0.50);
        let p90_ae = percentile_in_place(&mut self.abs_errs, 0.90);
        let wells = self.per_well.len();
        let mean_well_rmse = if wells > 0 {
            self.per_well.iter().map(|w| w.rmse).sum::<f64>() / wells as f64
        } else {
            f64::NAN
        };
        let rows_per_s = elapsed_s.and_then(|s| if s > 0.0 { Some(rows as f64 / s) } else { None });
        Metrics {
            rows,
            wells,
            rmse: if rows > 0 {
                (self.sum_sq / rows as f64).sqrt()
            } else {
                f64::NAN
            },
            mae: if rows > 0 {
                self.sum_abs / rows as f64
            } else {
                f64::NAN
            },
            bias: if rows > 0 {
                self.sum_err / rows as f64
            } else {
                f64::NAN
            },
            median_ae,
            p90_ae,
            mean_well_rmse,
            elapsed_s,
            rows_per_s,
            per_well: self.per_well,
        }
    }
}

fn main() {
    if let Err(err) = run() {
        eprintln!("error: {err}");
        std::process::exit(1);
    }
}

fn run() -> AnyResult<()> {
    let opts = parse_args()?;
    match opts.cmd.as_str() {
        "run-baseline" => run_baseline(&opts),
        "score-csv" => score_csv(&opts),
        "export" => export_validation(&opts),
        "help" | "-h" | "--help" => {
            print_help();
            Ok(())
        }
        other => Err(format!("unknown command {other:?}; run with --help").into()),
    }
}

fn parse_args() -> AnyResult<Opts> {
    let mut args: Vec<String> = env::args().skip(1).collect();
    if args.is_empty() {
        print_help();
        return Err("missing command".into());
    }
    if args[0] == "--help" || args[0] == "-h" {
        args[0] = "help".to_string();
    }
    let cmd = args.remove(0);
    let mut opts = Opts {
        cmd,
        train_dir: resolve_default_train_dir(),
        seed: 42,
        n_folds: 5,
        fold: 0,
        limit: 0,
        wells: None,
        strategy: "constant".to_string(),
        fallback_tvt: DEFAULT_FALLBACK_TVT,
        slope_cap: 0.001,
        slope_tail: 300,
        slope_min_points: 30,
        tail: 100,
        z_alpha: -1.0,
        z_slope_cap: 1.0,
        predictions_csv: None,
        truth_csv: None,
        export_dir: None,
        json: false,
        top: 12,
        threads: default_threads(),
    };

    let mut i = 0;
    while i < args.len() {
        let key = args[i].as_str();
        match key {
            "--json" => {
                opts.json = true;
                i += 1;
            }
            "--train-dir" => {
                opts.train_dir = PathBuf::from(take_value(&args, &mut i, key)?);
            }
            "--seed" => {
                opts.seed = take_value(&args, &mut i, key)?.parse()?;
            }
            "--n-folds" => {
                opts.n_folds = take_value(&args, &mut i, key)?.parse()?;
            }
            "--fold" => {
                opts.fold = take_value(&args, &mut i, key)?.parse()?;
            }
            "--limit" => {
                opts.limit = take_value(&args, &mut i, key)?.parse()?;
            }
            "--wells" => {
                let raw = take_value(&args, &mut i, key)?;
                let wells = raw
                    .split(',')
                    .map(str::trim)
                    .filter(|s| !s.is_empty())
                    .map(str::to_string)
                    .collect::<Vec<_>>();
                opts.wells = Some(wells);
            }
            "--strategy" => {
                opts.strategy = take_value(&args, &mut i, key)?;
            }
            "--fallback-tvt" => {
                opts.fallback_tvt = take_value(&args, &mut i, key)?.parse()?;
            }
            "--slope-cap" => {
                opts.slope_cap = take_value(&args, &mut i, key)?.parse()?;
            }
            "--slope-tail" => {
                opts.slope_tail = take_value(&args, &mut i, key)?.parse()?;
            }
            "--slope-min-points" => {
                opts.slope_min_points = take_value(&args, &mut i, key)?.parse()?;
            }
            "--tail" => {
                opts.tail = take_value(&args, &mut i, key)?.parse()?;
            }
            "--z-alpha" => {
                opts.z_alpha = take_value(&args, &mut i, key)?.parse()?;
            }
            "--z-slope-cap" => {
                opts.z_slope_cap = take_value(&args, &mut i, key)?.parse()?;
            }
            "--predictions-csv" => {
                opts.predictions_csv = Some(PathBuf::from(take_value(&args, &mut i, key)?));
            }
            "--truth-csv" => {
                opts.truth_csv = Some(PathBuf::from(take_value(&args, &mut i, key)?));
            }
            "--export-dir" => {
                opts.export_dir = Some(PathBuf::from(take_value(&args, &mut i, key)?));
            }
            "--top" => {
                opts.top = take_value(&args, &mut i, key)?.parse()?;
            }
            "--threads" => {
                opts.threads = take_value(&args, &mut i, key)?.parse()?;
            }
            _ => return Err(format!("unknown argument {key:?}").into()),
        }
    }
    if opts.n_folds == 0 {
        return Err("--n-folds must be > 0".into());
    }
    if opts.fold >= opts.n_folds {
        return Err(format!("--fold must be less than --n-folds ({})", opts.n_folds).into());
    }
    if opts.threads == 0 {
        return Err("--threads must be > 0".into());
    }
    Ok(opts)
}

fn take_value(args: &[String], i: &mut usize, key: &str) -> AnyResult<String> {
    if *i + 1 >= args.len() {
        return Err(format!("{key} needs a value").into());
    }
    let value = args[*i + 1].clone();
    *i += 2;
    Ok(value)
}

fn print_help() {
    eprintln!(
        "rogii-score\n\n\
Commands:\n\
  run-baseline --strategy constant|slope|tail-mean|tail-median|z-alpha|z-slope [--n-folds 5 --fold 0]\n\
  score-csv --truth-csv PATH --predictions-csv PATH\n\
  export --export-dir PATH [--n-folds 5 --fold 0]\n\n\
Shared options:\n\
  --train-dir PATH  --seed N  --n-folds N  --fold N  --limit N  --wells a,b,c  --threads N  --json\n\
  --threads defaults to min(available CPUs, 8), which is the local M1 Pro sweet spot.\n"
    );
}

fn resolve_default_train_dir() -> PathBuf {
    let p = PathBuf::from(DEFAULT_TRAIN_DIR);
    if p.exists() {
        p
    } else {
        PathBuf::from(DEFAULT_TRAIN_DIR_FROM_ROGII)
    }
}

fn default_threads() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get().min(8))
        .unwrap_or(1)
}

fn select_wells(opts: &Opts) -> AnyResult<Vec<String>> {
    let all = list_wells(&opts.train_dir)?;
    if let Some(wanted) = &opts.wells {
        let known = all.iter().cloned().collect::<HashSet<_>>();
        for w in wanted {
            if !known.contains(w) {
                return Err(format!("requested well {w:?} not found").into());
            }
        }
        let mut out = wanted.clone();
        if opts.limit > 0 && out.len() > opts.limit {
            out.truncate(opts.limit);
        }
        return Ok(out);
    }

    let mut wells = all;
    wells.sort_by_key(|w| stable_score(w, opts.seed));
    if opts.n_folds > 1 {
        wells = wells
            .into_iter()
            .enumerate()
            .filter_map(|(i, w)| {
                if i % opts.n_folds == opts.fold {
                    Some(w)
                } else {
                    None
                }
            })
            .collect();
    }
    if opts.limit > 0 && wells.len() > opts.limit {
        wells.truncate(opts.limit);
    }
    Ok(wells)
}

fn list_wells(train_dir: &Path) -> AnyResult<Vec<String>> {
    let mut wells = Vec::new();
    for entry in fs::read_dir(train_dir)? {
        let entry = entry?;
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if let Some(prefix) = name.strip_suffix("__horizontal_well.csv") {
            wells.push(prefix.to_string());
        }
    }
    wells.sort();
    if wells.is_empty() {
        return Err(format!("no horizontal train wells found in {}", train_dir.display()).into());
    }
    Ok(wells)
}

fn stable_score(well: &str, seed: u64) -> u64 {
    let payload = format!("{seed}:{well}");
    let mut hasher = Blake2bVar::new(8).expect("valid digest size");
    hasher.update(payload.as_bytes());
    let mut out = [0u8; 8];
    hasher
        .finalize_variable(&mut out)
        .expect("digest buffer fits");
    u64::from_be_bytes(out)
}

fn run_baseline(opts: &Opts) -> AnyResult<()> {
    let wells = select_wells(opts)?;
    let start = Instant::now();
    let mut acc = Accum::new();

    let pool = ThreadPoolBuilder::new().num_threads(opts.threads).build()?;
    let results = pool.install(|| {
        wells
            .par_iter()
            .map(|well| -> AnyResult<(String, Vec<f64>)> {
                let path = opts.train_dir.join(format!("{well}__horizontal_well.csv"));
                let data = read_horizontal(&path, well)?;
                let errs = match opts.strategy.as_str() {
                    "constant" => errors_constant(&data, opts.fallback_tvt),
                    "slope" => errors_slope(&data, opts),
                    "tail-mean" => errors_tail_mean(&data, opts),
                    "tail-median" => errors_tail_median(&data, opts),
                    "z-alpha" => errors_z_alpha(&data, opts),
                    "z-slope" => errors_z_slope(&data, opts),
                    other => return Err(format!("unknown strategy {other:?}").into()),
                };
                Ok((data.name, errs))
            })
            .collect::<Vec<_>>()
    });

    for result in results {
        let (well, errs) = result?;
        acc.add_well(well, &errs);
    }

    let metrics = acc.finish(Some(start.elapsed().as_secs_f64()));
    print_metrics(&metrics, opts);
    Ok(())
}

fn score_csv(opts: &Opts) -> AnyResult<()> {
    let predictions_path = opts
        .predictions_csv
        .as_ref()
        .ok_or("--predictions-csv is required")?;
    let truth = if let Some(truth_path) = &opts.truth_csv {
        read_truth_csv(truth_path)?
    } else {
        truth_from_train(opts)?
    };
    let preds = read_predictions(predictions_path)?;
    let mut by_well: HashMap<String, Vec<f64>> = HashMap::new();
    let mut missing = Vec::new();

    for row in &truth {
        match preds.get(&row.id) {
            Some(pred) if pred.is_finite() => {
                by_well
                    .entry(row.well.clone())
                    .or_default()
                    .push(*pred - row.tvt);
            }
            _ => {
                if missing.len() < 10 {
                    missing.push(row.id.clone());
                }
            }
        }
    }
    if !missing.is_empty() {
        return Err(format!("missing/non-finite predictions, first examples: {missing:?}").into());
    }

    let mut acc = Accum::new();
    let mut wells = by_well.into_iter().collect::<Vec<_>>();
    wells.sort_by(|a, b| a.0.cmp(&b.0));
    for (well, errs) in wells {
        acc.add_well(well, &errs);
    }
    let metrics = acc.finish(None);
    print_metrics(&metrics, opts);
    Ok(())
}

fn export_validation(opts: &Opts) -> AnyResult<()> {
    let export_dir = opts.export_dir.as_ref().ok_or("--export-dir is required")?;
    let test_dir = export_dir.join("test");
    fs::create_dir_all(&test_dir)?;
    let wells = select_wells(opts)?;
    let truth_path = export_dir.join("truth.csv");
    let sample_path = export_dir.join("sample_submission.csv");
    let selected_path = export_dir.join("selected_wells.txt");
    let mut truth = BufWriter::new(fs::File::create(&truth_path)?);
    let mut sample = BufWriter::new(fs::File::create(&sample_path)?);
    let mut selected = BufWriter::new(fs::File::create(&selected_path)?);
    writeln!(truth, "id,well,row,tvt")?;
    writeln!(sample, "id,tvt")?;

    let mut rows = 0usize;
    for well in wells {
        writeln!(selected, "{well}")?;
        let h_path = opts.train_dir.join(format!("{well}__horizontal_well.csv"));
        let t_path = opts.train_dir.join(format!("{well}__typewell.csv"));
        export_selected_columns(
            &h_path,
            &test_dir.join(format!("{well}__horizontal_well.csv")),
            &["MD", "X", "Y", "Z", "GR", "TVT_input"],
        )?;
        export_selected_columns(
            &t_path,
            &test_dir.join(format!("{well}__typewell.csv")),
            &["TVT", "GR"],
        )?;
        let data = read_horizontal(&h_path, &well)?;
        for i in 0..data.tvt.len() {
            if data.tvt[i].is_finite() && !data.tvt_input[i].is_finite() {
                let id = format!("{well}_{i}");
                writeln!(truth, "{id},{well},{i},{}", data.tvt[i])?;
                writeln!(sample, "{id},0.0")?;
                rows += 1;
            }
        }
    }
    truth.flush()?;
    sample.flush()?;
    selected.flush()?;
    println!(
        "{{\"export_dir\":\"{}\",\"test_dir\":\"{}\",\"truth_csv\":\"{}\",\"sample_submission_csv\":\"{}\",\"rows\":{}}}",
        export_dir.display(),
        test_dir.display(),
        truth_path.display(),
        sample_path.display(),
        rows
    );
    Ok(())
}

fn truth_from_train(opts: &Opts) -> AnyResult<Vec<TruthRow>> {
    let wells = select_wells(opts)?;
    let mut out = Vec::new();
    for well in wells {
        let path = opts.train_dir.join(format!("{well}__horizontal_well.csv"));
        let data = read_horizontal(&path, &well)?;
        for i in 0..data.tvt.len() {
            if data.tvt[i].is_finite() && !data.tvt_input[i].is_finite() {
                out.push(TruthRow {
                    id: format!("{well}_{i}"),
                    well: well.clone(),
                    tvt: data.tvt[i],
                });
            }
        }
    }
    Ok(out)
}

fn errors_constant(data: &WellData, fallback_tvt: f64) -> Vec<f64> {
    let last = last_known_tvt(data).unwrap_or(fallback_tvt);
    let mut errs = Vec::with_capacity(data.tvt.len() / 2);
    for i in 0..data.tvt.len() {
        if data.tvt[i].is_finite() && !data.tvt_input[i].is_finite() {
            errs.push(last - data.tvt[i]);
        }
    }
    errs
}

fn errors_slope(data: &WellData, opts: &Opts) -> Vec<f64> {
    let Some(last_i) = last_known_idx(data) else {
        return errors_constant(data, opts.fallback_tvt);
    };
    let last_tvt = data.tvt_input[last_i];
    let last_md = data.md[last_i];
    let slope = estimate_axis_slope(data, opts, last_i, Axis::Md, opts.slope_cap);
    let mut errs = Vec::with_capacity(data.tvt.len().saturating_sub(last_i));
    for i in 0..data.tvt.len() {
        if data.tvt[i].is_finite() && !data.tvt_input[i].is_finite() {
            let pred = last_tvt + slope * (data.md[i] - last_md);
            errs.push(pred - data.tvt[i]);
        }
    }
    errs
}

fn errors_tail_mean(data: &WellData, opts: &Opts) -> Vec<f64> {
    let pred = tail_values(data, opts.tail)
        .map(|vals| vals.iter().sum::<f64>() / vals.len() as f64)
        .unwrap_or(opts.fallback_tvt);
    errors_constant_value(data, pred)
}

fn errors_tail_median(data: &WellData, opts: &Opts) -> Vec<f64> {
    let pred = tail_values(data, opts.tail)
        .map(|mut vals| median_in_place(&mut vals))
        .unwrap_or(opts.fallback_tvt);
    errors_constant_value(data, pred)
}

fn errors_z_alpha(data: &WellData, opts: &Opts) -> Vec<f64> {
    let Some(last_i) = last_known_idx(data) else {
        return errors_constant(data, opts.fallback_tvt);
    };
    let last_tvt = data.tvt_input[last_i];
    let last_z = data.z[last_i];
    if !last_z.is_finite() {
        return errors_constant(data, opts.fallback_tvt);
    }
    let mut errs = Vec::with_capacity(data.tvt.len().saturating_sub(last_i));
    for i in 0..data.tvt.len() {
        if data.tvt[i].is_finite() && !data.tvt_input[i].is_finite() {
            let pred = if data.z[i].is_finite() {
                last_tvt + opts.z_alpha * (data.z[i] - last_z)
            } else {
                last_tvt
            };
            errs.push(pred - data.tvt[i]);
        }
    }
    errs
}

fn errors_z_slope(data: &WellData, opts: &Opts) -> Vec<f64> {
    let Some(last_i) = last_known_idx(data) else {
        return errors_constant(data, opts.fallback_tvt);
    };
    let last_tvt = data.tvt_input[last_i];
    let last_z = data.z[last_i];
    if !last_z.is_finite() {
        return errors_constant(data, opts.fallback_tvt);
    }
    let slope = estimate_axis_slope(data, opts, last_i, Axis::Z, opts.z_slope_cap);
    let mut errs = Vec::with_capacity(data.tvt.len().saturating_sub(last_i));
    for i in 0..data.tvt.len() {
        if data.tvt[i].is_finite() && !data.tvt_input[i].is_finite() {
            let pred = if data.z[i].is_finite() {
                last_tvt + slope * (data.z[i] - last_z)
            } else {
                last_tvt
            };
            errs.push(pred - data.tvt[i]);
        }
    }
    errs
}

fn errors_constant_value(data: &WellData, pred: f64) -> Vec<f64> {
    let mut errs = Vec::with_capacity(data.tvt.len() / 2);
    for i in 0..data.tvt.len() {
        if data.tvt[i].is_finite() && !data.tvt_input[i].is_finite() {
            errs.push(pred - data.tvt[i]);
        }
    }
    errs
}

fn tail_values(data: &WellData, tail: usize) -> Option<Vec<f64>> {
    let mut vals = Vec::with_capacity(tail.min(data.tvt_input.len()));
    for i in (0..data.tvt_input.len()).rev() {
        if data.tvt_input[i].is_finite() {
            vals.push(data.tvt_input[i]);
            if vals.len() >= tail {
                break;
            }
        }
    }
    if vals.is_empty() { None } else { Some(vals) }
}

fn last_known_idx(data: &WellData) -> Option<usize> {
    (0..data.tvt_input.len())
        .rev()
        .find(|&i| data.tvt_input[i].is_finite() && data.md[i].is_finite())
}

fn last_known_tvt(data: &WellData) -> Option<f64> {
    last_known_idx(data).map(|i| data.tvt_input[i])
}

enum Axis {
    Md,
    Z,
}

fn estimate_axis_slope(data: &WellData, opts: &Opts, last_i: usize, axis: Axis, cap: f64) -> f64 {
    let mut idx = Vec::new();
    for i in 0..=last_i {
        let x = match axis {
            Axis::Md => data.md[i],
            Axis::Z => data.z[i],
        };
        if data.tvt_input[i].is_finite() && x.is_finite() {
            idx.push(i);
        }
    }
    if idx.len() > opts.slope_tail {
        idx = idx[idx.len() - opts.slope_tail..].to_vec();
    }
    if idx.len() < opts.slope_min_points {
        return 0.0;
    }

    let mut slopes = Vec::with_capacity(idx.len() * (idx.len() - 1) / 2);
    for a in 0..idx.len() {
        let ia = idx[a];
        for &ib in &idx[a + 1..] {
            let xa = match axis {
                Axis::Md => data.md[ia],
                Axis::Z => data.z[ia],
            };
            let xb = match axis {
                Axis::Md => data.md[ib],
                Axis::Z => data.z[ib],
            };
            let dx = xb - xa;
            if dx.abs() > 1e-12 {
                slopes.push((data.tvt_input[ib] - data.tvt_input[ia]) / dx);
            }
        }
    }
    if slopes.is_empty() {
        return 0.0;
    }
    median_in_place(&mut slopes).clamp(-cap, cap)
}

fn read_horizontal(path: &Path, well: &str) -> AnyResult<WellData> {
    let file = fs::File::open(path)?;
    let row_capacity = estimate_row_capacity(&file);
    let mut reader = BufReader::with_capacity(256 * 1024, file);
    let mut header = String::new();
    reader.read_line(&mut header)?;
    let cols = split_csv_line(header.trim_end());
    let md_i = col_idx(&cols, "MD")?;
    let z_i = cols.iter().position(|&c| c == "Z");
    let tvt_i = col_idx(&cols, "TVT")?;
    let in_i = col_idx(&cols, "TVT_input")?;

    let mut md = Vec::with_capacity(row_capacity);
    let mut z = Vec::with_capacity(row_capacity);
    let mut tvt = Vec::with_capacity(row_capacity);
    let mut tvt_input = Vec::with_capacity(row_capacity);
    let mut line = String::new();
    loop {
        line.clear();
        if reader.read_line(&mut line)? == 0 {
            break;
        }
        let raw = line.trim_end();
        if raw.is_empty() {
            continue;
        }
        let parsed = parse_horizontal_fields(raw, md_i, z_i, tvt_i, in_i);
        md.push(parsed.0);
        z.push(parsed.1);
        tvt.push(parsed.2);
        tvt_input.push(parsed.3);
    }
    Ok(WellData {
        name: well.to_string(),
        md,
        z,
        tvt,
        tvt_input,
    })
}

fn estimate_row_capacity(file: &fs::File) -> usize {
    file.metadata()
        .ok()
        .map(|m| ((m.len() as usize) / 96).clamp(1024, 16_384))
        .unwrap_or(4096)
}

fn parse_horizontal_fields(
    line: &str,
    md_i: usize,
    z_i: Option<usize>,
    tvt_i: usize,
    in_i: usize,
) -> (f64, f64, f64, f64) {
    let mut md = f64::NAN;
    let mut z = f64::NAN;
    let mut tvt = f64::NAN;
    let mut tvt_input = f64::NAN;
    let bytes = line.as_bytes();
    let mut col = 0usize;
    let mut start = 0usize;
    let mut i = 0usize;
    while i <= bytes.len() {
        if i == bytes.len() || bytes[i] == b',' {
            if col == md_i {
                md = parse_field(&line[start..i]);
            } else if Some(col) == z_i {
                z = parse_field(&line[start..i]);
            } else if col == tvt_i {
                tvt = parse_field(&line[start..i]);
            } else if col == in_i {
                tvt_input = parse_field(&line[start..i]);
            }
            col += 1;
            start = i + 1;
        }
        i += 1;
    }
    (md, z, tvt, tvt_input)
}

fn read_truth_csv(path: &Path) -> AnyResult<Vec<TruthRow>> {
    let file = fs::File::open(path)?;
    let mut reader = BufReader::new(file);
    let mut header = String::new();
    reader.read_line(&mut header)?;
    let cols = split_csv_line(header.trim_end());
    let id_i = col_idx(&cols, "id")?;
    let well_i = col_idx(&cols, "well")?;
    let tvt_i = col_idx(&cols, "tvt")?;
    let mut out = Vec::new();
    let mut line = String::new();
    loop {
        line.clear();
        if reader.read_line(&mut line)? == 0 {
            break;
        }
        let raw = line.trim_end();
        if raw.is_empty() {
            continue;
        }
        let fields = split_csv_line(raw);
        out.push(TruthRow {
            id: fields.get(id_i).copied().unwrap_or("").to_string(),
            well: fields.get(well_i).copied().unwrap_or("").to_string(),
            tvt: parse_field(fields.get(tvt_i).copied().unwrap_or("")),
        });
    }
    Ok(out)
}

fn read_predictions(path: &Path) -> AnyResult<HashMap<String, f64>> {
    let file = fs::File::open(path)?;
    let mut reader = BufReader::new(file);
    let mut header = String::new();
    reader.read_line(&mut header)?;
    let cols = split_csv_line(header.trim_end());
    let id_i = col_idx(&cols, "id")?;
    let tvt_i = col_idx(&cols, "tvt")?;
    let mut out = HashMap::new();
    let mut line = String::new();
    loop {
        line.clear();
        if reader.read_line(&mut line)? == 0 {
            break;
        }
        let raw = line.trim_end();
        if raw.is_empty() {
            continue;
        }
        let fields = split_csv_line(raw);
        let id = fields.get(id_i).copied().unwrap_or("").to_string();
        if id.is_empty() {
            continue;
        }
        if out
            .insert(
                id.clone(),
                parse_field(fields.get(tvt_i).copied().unwrap_or("")),
            )
            .is_some()
        {
            return Err(format!("duplicate prediction id {id:?}").into());
        }
    }
    Ok(out)
}

fn export_selected_columns(input: &Path, output: &Path, wanted: &[&str]) -> AnyResult<()> {
    let file = fs::File::open(input)?;
    let mut reader = BufReader::new(file);
    let mut header = String::new();
    reader.read_line(&mut header)?;
    let cols = split_csv_line(header.trim_end());
    let indices = wanted
        .iter()
        .map(|name| col_idx(&cols, name))
        .collect::<Result<Vec<_>, _>>()?;
    let out_file = fs::File::create(output)?;
    let mut writer = BufWriter::new(out_file);
    writeln!(writer, "{}", wanted.join(","))?;
    let mut line = String::new();
    loop {
        line.clear();
        if reader.read_line(&mut line)? == 0 {
            break;
        }
        let raw = line.trim_end();
        if raw.is_empty() {
            continue;
        }
        let fields = split_csv_line(raw);
        for (k, &idx) in indices.iter().enumerate() {
            if k > 0 {
                write!(writer, ",")?;
            }
            write!(writer, "{}", fields.get(idx).copied().unwrap_or(""))?;
        }
        writeln!(writer)?;
    }
    writer.flush()?;
    Ok(())
}

fn split_csv_line(line: &str) -> Vec<&str> {
    line.trim_end_matches('\r').split(',').collect()
}

fn col_idx(cols: &[&str], name: &str) -> AnyResult<usize> {
    cols.iter()
        .position(|&c| c == name)
        .ok_or_else(|| format!("missing column {name:?}").into())
}

fn parse_field(raw: &str) -> f64 {
    let s = raw.trim();
    if s.is_empty() || matches!(s, "NA" | "NaN" | "nan" | "null") {
        f64::NAN
    } else {
        s.parse::<f64>().unwrap_or(f64::NAN)
    }
}

fn percentile_in_place(xs: &mut [f64], p: f64) -> f64 {
    if xs.is_empty() {
        return f64::NAN;
    }
    let idx = ((xs.len() - 1) as f64 * p).round() as usize;
    let idx = idx.min(xs.len() - 1);
    let (_lo, value, _hi) =
        xs.select_nth_unstable_by(idx, |a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    *value
}

fn median_in_place(xs: &mut [f64]) -> f64 {
    xs.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    let n = xs.len();
    if n % 2 == 1 {
        xs[n / 2]
    } else {
        0.5 * (xs[n / 2 - 1] + xs[n / 2])
    }
}

fn print_metrics(metrics: &Metrics, opts: &Opts) {
    if opts.json {
        print_metrics_json(metrics);
        return;
    }
    println!(
        "rows={} wells={} rmse={:.4} mae={:.4} bias={:.4} median_ae={:.4} p90_ae={:.4} mean_well_rmse={:.4}",
        metrics.rows,
        metrics.wells,
        metrics.rmse,
        metrics.mae,
        metrics.bias,
        metrics.median_ae,
        metrics.p90_ae,
        metrics.mean_well_rmse,
    );
    if let Some(elapsed) = metrics.elapsed_s {
        if let Some(rate) = metrics.rows_per_s {
            println!("elapsed_s={elapsed:.6} rows_per_s={rate:.1}");
        } else {
            println!("elapsed_s={elapsed:.6}");
        }
    }
    let mut worst = metrics.per_well.clone();
    worst.sort_by(|a, b| b.rmse.partial_cmp(&a.rmse).unwrap_or(Ordering::Equal));
    if !worst.is_empty() && opts.top > 0 {
        println!("\nWorst wells:");
        println!(
            "{:>12} {:>8} {:>12} {:>12} {:>12}",
            "well", "rows", "rmse", "mae", "bias"
        );
        for w in worst.iter().take(opts.top) {
            println!(
                "{:>12} {:>8} {:>12.6} {:>12.6} {:>12.6}",
                w.well, w.rows, w.rmse, w.mae, w.bias
            );
        }
    }
}

fn print_metrics_json(metrics: &Metrics) {
    println!(
        "{{\"rows\":{},\"wells\":{},\"rmse\":{},\"mae\":{},\"bias\":{},\"median_ae\":{},\"p90_ae\":{},\"mean_well_rmse\":{},\"elapsed_s\":{},\"rows_per_s\":{}}}",
        metrics.rows,
        metrics.wells,
        json_f(metrics.rmse),
        json_f(metrics.mae),
        json_f(metrics.bias),
        json_f(metrics.median_ae),
        json_f(metrics.p90_ae),
        json_f(metrics.mean_well_rmse),
        metrics
            .elapsed_s
            .map(json_f)
            .unwrap_or_else(|| "null".to_string()),
        metrics
            .rows_per_s
            .map(json_f)
            .unwrap_or_else(|| "null".to_string()),
    );
}

fn json_f(x: f64) -> String {
    if x.is_finite() {
        format!("{x}")
    } else {
        "null".to_string()
    }
}
