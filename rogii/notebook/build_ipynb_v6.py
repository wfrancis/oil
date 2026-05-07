"""Build a Jupyter .ipynb file for v6 exact lookup with residual fallback."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
source = (ROOT / "kaggle_cell_v6.py").read_text()

notebook = {
    "cells": [
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [line + "\n" for line in source.splitlines()],
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
            "version": "3.11",
            "mimetype": "text/x-python",
            "codemirror_mode": {"name": "ipython", "version": 3},
            "pygments_lexer": "ipython3",
            "nbconvert_exporter": "python",
            "file_extension": ".py",
        },
        "kaggle": {
            "accelerator": "none",
            "dataSources": [],
            "dockerImageVersionId": 30787,
            "isGpuEnabled": False,
            "isInternetEnabled": False,
            "language": "python",
            "sourceType": "notebook",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

(ROOT / "submission.ipynb").write_text(json.dumps(notebook, indent=1))
print(f"Wrote {ROOT / 'submission.ipynb'} from kaggle_cell_v6.py")
