"""Download KazSAnDRA (CC BY 4.0) from GitHub at a pinned commit.

    python -m src.download_data                  # saves the CSVs in data/raw/
    python -m src.download_data --from-local DIR # use a local clone instead

Only needed to rebuild the labeling pool. Training uses the committed data/ files.
"""
import argparse
import hashlib
import io
import shutil
import urllib.request
import zipfile
from pathlib import Path

from .utils import load_config, path

URL = "https://github.com/IS2AI/KazSAnDRA/raw/{commit}/dataset/{name}"
SHA256 = {  # checksums of the ZIP files at the pinned commit
    "06_sc_train_ib.zip": "70ff171a8d600c6ce89597cb68b968fb1d500a350a3c84a92879c6a0a0fabd5e",
    "09_sc_valid.zip": "06031177d1423d4bb44a459e5bf051aa6daac222fa80f022dd6f2a929a362005",
    "10_sc_test.zip": "f1456fdda61257b5dc5eb26d86b39273e627fa30cd7b1bb63f5f9795dd34f57e",
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from-local", type=Path, help="path to a KazSAnDRA clone (uses its dataset/ folder)")
    args = ap.parse_args()
    cfg = load_config()
    out = path(cfg, "raw_dir")
    out.mkdir(parents=True, exist_ok=True)
    for name in cfg["kazsandra"]["files"]:
        if args.from_local:
            data = (args.from_local / "dataset" / name).read_bytes()
        else:
            url = URL.format(commit=cfg["kazsandra"]["commit"], name=name)
            print("downloading", url)
            with urllib.request.urlopen(url) as r:
                data = r.read()
        digest = hashlib.sha256(data).hexdigest()
        if SHA256.get(name) and digest != SHA256[name]:
            raise ValueError(f"{name}: checksum {digest} does not match the pinned file")
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            for member in z.namelist():
                if member.endswith(".csv"):
                    target = out / Path(member).name
                    with z.open(member) as src, open(target, "wb") as dst:
                        shutil.copyfileobj(src, dst)
                    print(f"  {name} (sha256 {digest[:12]}...) -> {target}")


if __name__ == "__main__":
    main()
