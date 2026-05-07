"""Build a Jupyter .ipynb file containing the assembled Kaggle cell.

The .ipynb is a single JSON document with one code cell holding the entire
submission pipeline (modules + run logic).
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CELL = (ROOT / "notebook" / "kaggle_cell.py").read_text()
OUT = ROOT / "notebook" / "submission.ipynb"

notebook = {
    "cells": [
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": CELL.splitlines(keepends=True),
        }
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.10",
        },
        "kaggle": {
            "accelerator": "none",
            "dataSources": [
                {
                    "sourceId": "rogii-wellbore-geology-prediction",
                    "sourceType": "competition",
                }
            ],
            "isInternetEnabled": False,
            "language": "python",
            "sourceType": "notebook",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUT.write_text(json.dumps(notebook, indent=1))
print(f"Wrote {OUT} ({OUT.stat().st_size:,} bytes)")
