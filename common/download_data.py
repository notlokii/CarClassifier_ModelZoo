"""Download the Kaggle 'Car Body Types Images' dataset into data/raw/.

Prereqs (one-time):
  1. pip install kaggle           (already in requirements.txt)
  2. Kaggle API token at ~/.kaggle/kaggle.json  (chmod 600)

Usage:
  python common/download_data.py
"""

import subprocess
import sys
from pathlib import Path

DATASET = "ademboukhris/cars-body-type-cropped"
RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    token = Path.home() / ".kaggle" / "kaggle.json"
    if not token.exists():
        sys.exit(
            f"Kaggle API token not found at {token}\n"
            "Create one at https://www.kaggle.com/settings/account -> 'Create New Token',\n"
            "then place kaggle.json there and run: chmod 600 ~/.kaggle/kaggle.json"
        )

    print(f"Downloading {DATASET} -> {RAW_DIR} ...")
    subprocess.run(
        [
            "kaggle", "datasets", "download",
            "-d", DATASET,
            "-p", str(RAW_DIR),
            "--unzip",
        ],
        check=True,
    )
    print("Done. Contents of data/raw/:")
    for p in sorted(RAW_DIR.iterdir()):
        print("  ", p.name)


if __name__ == "__main__":
    main()
