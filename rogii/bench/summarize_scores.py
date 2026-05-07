"""Summarize fold-level JSON metrics from local_score.py or score-rust."""

from __future__ import annotations

import argparse
import glob
import json
import math
from pathlib import Path
from typing import Any


def _expand(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        if matches:
            paths.extend(Path(p) for p in matches)
        else:
            paths.append(Path(pattern))
    return paths


def _load(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def summarize(paths: list[Path]) -> dict[str, Any]:
    rows = 0
    wells = 0
    sq_err = 0.0
    rmses: list[float] = []
    maes: list[float] = []
    biases: list[float] = []
    elapsed = 0.0
    fold_rows = []

    for path in paths:
        m = _load(path)
        n = int(m["rows"])
        rmse = float(m["rmse"])
        rows += n
        wells += int(m.get("wells", 0))
        sq_err += rmse * rmse * n
        rmses.append(rmse)
        maes.append(float(m.get("mae", math.nan)))
        biases.append(float(m.get("bias", math.nan)))
        if m.get("elapsed_s") is not None:
            elapsed += float(m["elapsed_s"])
        fold_rows.append(
            {
                "path": str(path),
                "rows": n,
                "wells": int(m.get("wells", 0)),
                "rmse": rmse,
                "mae": float(m.get("mae", math.nan)),
                "bias": float(m.get("bias", math.nan)),
            }
        )

    row_weighted_rmse = math.sqrt(sq_err / rows) if rows else math.nan
    return {
        "files": len(paths),
        "rows": rows,
        "wells": wells,
        "row_weighted_rmse": row_weighted_rmse,
        "mean_fold_rmse": sum(rmses) / len(rmses) if rmses else math.nan,
        "mean_fold_mae": sum(maes) / len(maes) if maes else math.nan,
        "mean_fold_bias": sum(biases) / len(biases) if biases else math.nan,
        "elapsed_s_sum": elapsed if elapsed else None,
        "folds": fold_rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metrics_json", nargs="+", help="JSON files or shell-style globs.")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    paths = _expand(args.metrics_json)
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing metric files: {missing[:10]}")

    out = summarize(paths)
    if args.json:
        print(json.dumps(out, indent=2, sort_keys=True))
    else:
        print(
            "files={files} rows={rows} wells={wells} "
            "row_weighted_rmse={row_weighted_rmse:.4f} "
            "mean_fold_rmse={mean_fold_rmse:.4f} "
            "mean_fold_mae={mean_fold_mae:.4f} "
            "mean_fold_bias={mean_fold_bias:.4f}".format(**out)
        )
        if out["elapsed_s_sum"] is not None:
            print(f"elapsed_s_sum={out['elapsed_s_sum']:.3f}")
        print("\nFolds:")
        for fold in out["folds"]:
            print(
                "{path} rows={rows} wells={wells} rmse={rmse:.4f} "
                "mae={mae:.4f} bias={bias:.4f}".format(**fold)
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
