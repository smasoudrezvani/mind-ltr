"""Download MINDsmall (train + dev) into data/raw/.

Usage:  python scripts/download_mind.py

MINDsmall: 50k users sampled from MINDlarge — plenty for this project and
trains in minutes on a laptop. Swap URLs for MINDlarge later if you want.
Dataset page + license: https://msnews.github.io/
"""

from __future__ import annotations

import urllib.request
import zipfile
from pathlib import Path

URLS = {
    "MINDsmall_train": "https://mind201910small.blob.core.windows.net/release/MINDsmall_train.zip",
    "MINDsmall_dev": "https://mind201910small.blob.core.windows.net/release/MINDsmall_dev.zip",
}

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    for name, url in URLS.items():
        out_dir = RAW / name
        if (out_dir / "behaviors.tsv").exists():
            print(f"[skip] {name} already present")
            continue
        zip_path = RAW / f"{name}.zip"
        print(f"[down] {url}")
        urllib.request.urlretrieve(url, zip_path)
        print(f"[unzip] -> {out_dir}")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(out_dir)
        zip_path.unlink()
    print("done.")


if __name__ == "__main__":
    main()
