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
    # Microsoft's original Azure blob (mind201910small.blob.core.windows.net)
    # returns HTTP 409 since mid-2024 (public access disabled). The
    # recommenders-team maintains an official mirror on Hugging Face; these
    # URLs are what recommenders/datasets/mind.py itself now uses.
    "MINDsmall_train": "https://huggingface.co/datasets/Recommenders/MIND/resolve/main/MINDsmall_train.zip",
    "MINDsmall_dev": "https://huggingface.co/datasets/Recommenders/MIND/resolve/main/MINDsmall_dev.zip",
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
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as resp, open(zip_path, "wb") as out:
            while chunk := resp.read(1 << 20):
                out.write(chunk)
        print(f"[unzip] -> {out_dir}")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(out_dir)
        zip_path.unlink()
    print("done.")


if __name__ == "__main__":
    main()
