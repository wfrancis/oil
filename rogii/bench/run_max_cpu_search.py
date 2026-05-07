"""Run an 8-hour submit-proof local residual search.

This script deliberately does not call Kaggle. It evaluates local 5-fold
residual-model candidates, records every completed candidate into the submit
guard ledger, and keeps the machine busy until the requested wall-time target.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = Path("/tmp/rogii_max_cpu_search")
CURRENT_BEST = 14.5861


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def candidate_stream() -> list[dict[str, Any]]:
    seeds = [20260506, 20260507, 20260508, 20260509]
    train_rows = [1_200_000, 2_000_000, 800_000, 3_000_000]
    shrinks = [0.65, 0.75, 0.85, 0.55, 0.95, 0.45, 1.05]
    n_estimators = [700, 1000, 500, 1400]
    learning_rates = [0.035, 0.025, 0.05, 0.018]
    leaves = [63, 31, 127]
    min_child = [200, 100, 400]
    subsamples = [0.9, 1.0, 0.8]
    colsamples = [0.9, 1.0, 0.8]
    lambdas = [2.0, 5.0, 0.5, 10.0]

    anchor = {
        "name": "v5_control",
        "model_seed": 20260506,
        "train_rows": 1_200_000,
        "shrink": 0.75,
        "n_estimators": 700,
        "learning_rate": 0.035,
        "num_leaves": 63,
        "min_child_samples": 200,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "reg_lambda": 2.0,
    }
    out = [anchor]

    idx = 0
    for values in itertools.product(
        seeds,
        train_rows,
        shrinks,
        n_estimators,
        learning_rates,
        leaves,
        min_child,
        subsamples,
        colsamples,
        lambdas,
    ):
        (
            seed,
            rows,
            shrink,
            trees,
            lr,
            num_leaves,
            min_child_samples,
            subsample,
            colsample,
            reg_lambda,
        ) = values
        # Keep the search broad but sane: pair bigger tree counts with smaller LR
        # more often, and skip obviously duplicate control.
        if trees >= 1000 and lr >= 0.05:
            continue
        if trees <= 500 and lr <= 0.018:
            continue
        cand = {
            "name": f"cand_{idx:05d}",
            "model_seed": seed,
            "train_rows": rows,
            "shrink": shrink,
            "n_estimators": trees,
            "learning_rate": lr,
            "num_leaves": num_leaves,
            "min_child_samples": min_child_samples,
            "subsample": subsample,
            "colsample_bytree": colsample,
            "reg_lambda": reg_lambda,
        }
        if any(
            cand[k] != anchor[k]
            for k in cand
            if k in anchor and k not in {"name"}
        ):
            out.append(cand)
            idx += 1
    return out


def run_fold(candidate: dict[str, Any], fold: int, out_dir: Path, threads: int) -> Path:
    out_path = out_dir / f"fold_{fold}.json"
    cmd = [
        sys.executable,
        str(ROOT / "bench" / "local_score.py"),
        "run-residual",
        "--n-folds",
        "5",
        "--fold",
        str(fold),
        "--train-rows",
        str(candidate["train_rows"]),
        "--shrink",
        str(candidate["shrink"]),
        "--model-seed",
        str(candidate["model_seed"]),
        "--threads",
        str(threads),
        "--n-estimators",
        str(candidate["n_estimators"]),
        "--learning-rate",
        str(candidate["learning_rate"]),
        "--num-leaves",
        str(candidate["num_leaves"]),
        "--min-child-samples",
        str(candidate["min_child_samples"]),
        "--subsample",
        str(candidate["subsample"]),
        "--colsample-bytree",
        str(candidate["colsample_bytree"]),
        "--reg-lambda",
        str(candidate["reg_lambda"]),
        "--json",
    ]
    env = os.environ.copy()
    env.setdefault("OMP_NUM_THREADS", str(threads))
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    with out_path.open("w") as f:
        subprocess.run(cmd, cwd=ROOT.parent, env=env, stdout=f, stderr=subprocess.STDOUT, check=True)
    return out_path


def load_metric(path: Path) -> dict[str, Any]:
    text = path.read_text()
    start = text.find("{")
    if start < 0:
        raise ValueError(f"No JSON object in {path}")
    return json.loads(text[start:])


def summarize(paths: list[Path]) -> dict[str, Any]:
    rows = 0
    sq = 0.0
    folds = []
    for path in sorted(paths):
        m = load_metric(path)
        n = int(m["rows"])
        rows += n
        sq += float(m["rmse"]) ** 2 * n
        folds.append(
            {
                "path": str(path),
                "rows": n,
                "rmse": float(m["rmse"]),
                "mae": float(m.get("mae", math.nan)),
                "bias": float(m.get("bias", math.nan)),
            }
        )
    return {
        "rows": rows,
        "row_weighted_rmse": math.sqrt(sq / rows) if rows else math.nan,
        "mean_fold_rmse": sum(f["rmse"] for f in folds) / len(folds),
        "folds": folds,
    }


def record_guard(label: str, wall_seconds: float, note: str) -> None:
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "bench" / "submit_guard.py"),
            "record",
            "--label",
            label,
            "--wall-seconds",
            f"{wall_seconds:.3f}",
            "--note",
            note,
        ],
        cwd=ROOT.parent,
        check=False,
    )


def evaluate_candidate(
    candidate: dict[str, Any],
    root: Path,
    parallel_folds: int,
    threads_per_fold: int,
) -> dict[str, Any]:
    out_dir = root / candidate["name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "candidate.json").write_text(json.dumps(candidate, indent=2, sort_keys=True) + "\n")
    started = time.perf_counter()
    paths: list[Path] = []
    with ThreadPoolExecutor(max_workers=parallel_folds) as pool:
        futures = [
            pool.submit(run_fold, candidate, fold, out_dir, threads_per_fold)
            for fold in range(5)
        ]
        for future in as_completed(futures):
            paths.append(future.result())
    wall_seconds = time.perf_counter() - started
    summary = summarize(paths)
    result = {
        "event": "candidate_complete",
        "utc": utc_now(),
        "candidate": candidate,
        "wall_seconds": wall_seconds,
        **summary,
        "delta_vs_current_best": summary["row_weighted_rmse"] - CURRENT_BEST,
    }
    (out_dir / "summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-seconds", type=float, default=8 * 3600)
    parser.add_argument("--parallel-folds", type=int, default=5)
    parser.add_argument("--threads-per-fold", type=int, default=2)
    parser.add_argument("--out-dir", default=str(OUT_ROOT))
    args = parser.parse_args()

    root = Path(args.out_dir)
    root.mkdir(parents=True, exist_ok=True)
    log_path = root / "search.log.jsonl"
    best_path = root / "best.json"
    started = time.perf_counter()
    best: dict[str, Any] | None = None
    candidates = candidate_stream()
    print(
        f"{utc_now()} starting max CPU search target={args.target_seconds}s "
        f"parallel_folds={args.parallel_folds} threads_per_fold={args.threads_per_fold} "
        f"candidates={len(candidates)} out={root}",
        flush=True,
    )

    i = 0
    while True:
        candidate = dict(candidates[i % len(candidates)])
        if i >= len(candidates):
            candidate["name"] = f"{candidate['name']}_cycle{i // len(candidates):03d}"
            candidate["model_seed"] = int(candidate["model_seed"]) + i
        print(f"{utc_now()} candidate_start {candidate['name']} {candidate}", flush=True)
        try:
            result = evaluate_candidate(candidate, root, args.parallel_folds, args.threads_per_fold)
        except Exception as exc:
            wall = time.perf_counter() - started
            record = {
                "event": "candidate_failed",
                "utc": utc_now(),
                "candidate": candidate,
                "elapsed_since_start": wall,
                "error": repr(exc),
            }
            with log_path.open("a") as f:
                f.write(json.dumps(record, sort_keys=True) + "\n")
            print(json.dumps(record, sort_keys=True), flush=True)
            i += 1
            continue

        with log_path.open("a") as f:
            f.write(json.dumps(result, sort_keys=True) + "\n")
        record_guard(
            result["candidate"]["name"],
            float(result["wall_seconds"]),
            "max_cpu_local_residual_search_no_submit",
        )
        if best is None or result["row_weighted_rmse"] < best["row_weighted_rmse"]:
            best = result
            best_path.write_text(json.dumps(best, indent=2, sort_keys=True) + "\n")
            print(
                f"{utc_now()} new_best rmse={best['row_weighted_rmse']:.6f} "
                f"delta={best['delta_vs_current_best']:+.6f} name={best['candidate']['name']}",
                flush=True,
            )
        else:
            print(
                f"{utc_now()} candidate_done rmse={result['row_weighted_rmse']:.6f} "
                f"delta={result['delta_vs_current_best']:+.6f} name={result['candidate']['name']}",
                flush=True,
            )

        i += 1
        if time.perf_counter() - started >= args.target_seconds:
            break

    print(f"{utc_now()} finished target reached best={best_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
