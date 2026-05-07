"""Fail-closed guard for Kaggle submissions.

The next ROGII submission requires at least 8 hours of active search wall time
recorded after the latest submission. This script tracks that ledger and exits
non-zero until the requirement is met.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
STATE_PATH = ROOT / "submit_guard_state.json"
LOG_PATH = ROOT / "search_runs.jsonl"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def load_state() -> dict[str, Any]:
    if not STATE_PATH.exists():
        raise FileNotFoundError(f"Missing guard state: {STATE_PATH}")
    return json.loads(STATE_PATH.read_text())


def save_state(state: dict[str, Any]) -> None:
    STATE_PATH.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")


def append_log(record: dict[str, Any]) -> None:
    with LOG_PATH.open("a") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")


def status_payload() -> dict[str, Any]:
    state = load_state()
    have = float(state.get("accumulated_wall_seconds", 0.0))
    need = float(state.get("min_wall_seconds_before_submit", 8 * 3600))
    remaining = max(0.0, need - have)
    return {
        **state,
        "remaining_wall_seconds": remaining,
        "remaining_wall_hours": remaining / 3600.0,
        "ready_to_submit": have >= need,
    }


def cmd_status(args: argparse.Namespace) -> int:
    payload = status_payload()
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(
            "submit_ready={ready_to_submit} accumulated_hours={have:.3f} "
            "required_hours={need:.3f} remaining_hours={remaining:.3f}".format(
                ready_to_submit=payload["ready_to_submit"],
                have=float(payload["accumulated_wall_seconds"]) / 3600.0,
                need=float(payload["min_wall_seconds_before_submit"]) / 3600.0,
                remaining=payload["remaining_wall_hours"],
            )
        )
        print(payload.get("hard_constraint", ""))
    return 0


def cmd_check(args: argparse.Namespace) -> int:
    payload = status_payload()
    if payload["ready_to_submit"]:
        print(
            "OK: submit guard passed with %.3f recorded active-search hours."
            % (float(payload["accumulated_wall_seconds"]) / 3600.0)
        )
        return 0
    print(
        "BLOCKED: submit guard requires 8.000 active-search hours; "
        "%.3f recorded, %.3f remaining."
        % (
            float(payload["accumulated_wall_seconds"]) / 3600.0,
            payload["remaining_wall_hours"],
        )
    )
    return 2


def cmd_record(args: argparse.Namespace) -> int:
    if args.wall_seconds <= 0:
        raise ValueError("--wall-seconds must be positive")
    state = load_state()
    state["accumulated_wall_seconds"] = float(state.get("accumulated_wall_seconds", 0.0)) + float(
        args.wall_seconds
    )
    state["last_recorded_utc"] = utc_now()
    record = {
        "event": "search_run_recorded",
        "utc": state["last_recorded_utc"],
        "label": args.label,
        "wall_seconds": float(args.wall_seconds),
        "cpu_seconds": float(args.cpu_seconds),
        "note": args.note,
        "accumulated_wall_seconds_after": state["accumulated_wall_seconds"],
    }
    append_log(record)
    save_state(state)
    return cmd_status(argparse.Namespace(json=False))


def cmd_reset(args: argparse.Namespace) -> int:
    state = load_state()
    state["last_submission_utc"] = args.last_submission_utc
    state["accumulated_wall_seconds"] = 0
    state["last_recorded_utc"] = utc_now()
    append_log(
        {
            "event": "guard_cycle_reset",
            "utc": state["last_recorded_utc"],
            "last_submission_utc": args.last_submission_utc,
            "wall_seconds": 0,
            "cpu_seconds": 0,
            "note": args.note,
        }
    )
    save_state(state)
    return cmd_status(argparse.Namespace(json=False))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    status = sub.add_parser("status")
    status.add_argument("--json", action="store_true")
    status.set_defaults(func=cmd_status)

    check = sub.add_parser("check")
    check.set_defaults(func=cmd_check)

    record = sub.add_parser("record")
    record.add_argument("--label", required=True)
    record.add_argument("--wall-seconds", type=float, required=True)
    record.add_argument("--cpu-seconds", type=float, default=0.0)
    record.add_argument("--note", default="")
    record.set_defaults(func=cmd_record)

    reset = sub.add_parser("reset-cycle")
    reset.add_argument("--last-submission-utc", required=True)
    reset.add_argument("--note", default="")
    reset.set_defaults(func=cmd_reset)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
