"""Lossless letterbox cache and original-image integrity audit (train/valid only)."""
import hashlib
import io
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
from PIL import Image
from utils import DATA_DIR, PROJECT_DIR, IMAGE_SIZE, letterbox, load_images

CACHE_DIR = PROJECT_DIR / "work" / "cache384"

def decode(path):
    raw = Path(path).read_bytes()
    with Image.open(io.BytesIO(raw)) as image:
        size = image.size
        arr = np.array(letterbox(image), dtype=np.uint8)
    return arr, hashlib.sha256(raw).hexdigest(), size

def prepare():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    all_hashes = {}
    report = {"image_size": IMAGE_SIZE, "test_used_for_training": False, "splits": {}}
    for split in ("train", "valid"):
        paths, labels = load_images(DATA_DIR / f"{split}.csv", DATA_DIR / split)
        target, manifest = CACHE_DIR / f"{split}.npy", CACHE_DIR / f"{split}.json"
        if target.exists() and manifest.exists():
            info = json.loads(manifest.read_text(encoding="utf-8"))
            if info["names"] != [p.name for p in paths] or info["labels"] != labels:
                raise ValueError("Cache manifest differs from source CSV; rebuild cache")
            with ThreadPoolExecutor(max_workers=12) as pool:
                hashes = list(pool.map(lambda p: hashlib.sha256(p.read_bytes()).hexdigest(), paths))
            if hashes != info["sha256"]:
                raise ValueError("Source images changed; rebuild cache")
        else:
            temp = target.with_suffix(".tmp.npy")
            images = np.lib.format.open_memmap(temp, mode="w+", dtype=np.uint8, shape=(len(paths), IMAGE_SIZE, IMAGE_SIZE, 3))
            hashes, sizes = [], []
            with ThreadPoolExecutor(max_workers=12) as pool:
                for start in range(0, len(paths), 256):
                    for offset, (array, digest, size) in enumerate(pool.map(decode, paths[start:start+256])):
                        images[start + offset] = array
                        hashes.append(digest)
                        sizes.append(size)
                    if start % 2560 == 0:
                        print(f"cache {split}: {min(start+256, len(paths))}/{len(paths)}", flush=True)
            images.flush()
            del images
            temp.replace(target)
            info = {"names": [p.name for p in paths], "labels": labels, "sha256": hashes,
                    "original_sizes": sizes, "transform": "RGB LANCZOS thumbnail 384 + centered black padding"}
            manifest.write_text(json.dumps(info), encoding="utf-8")
        all_hashes[split] = info["sha256"]
        report["splits"][split] = {"count": len(labels), "positive": sum(labels), "negative": len(labels)-sum(labels), "decoded": True,
             "csv_sha256": hashlib.sha256((DATA_DIR / f"{split}.csv").read_bytes()).hexdigest()}
    valid_hashes = set(all_hashes["valid"])
    excluded = [i for i, h in enumerate(all_hashes["train"]) if h in valid_hashes]
    report["train_indices_excluded_exact_validation_duplicates"] = excluded
    report["train_count_after_exclusion"] = len(all_hashes["train"])-len(excluded)
    report["patient_level_leakage"] = "Cannot verify: no patient identifiers supplied. Official split retained."
    (CACHE_DIR / "audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    return report

if __name__ == "__main__":
    prepare()
