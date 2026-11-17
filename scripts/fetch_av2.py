#!/usr/bin/env python3
"""Download a lidar-only subset of the Argoverse 2 Sensor dataset (val split).

Fetches, for the first N logs in sorted order (deterministic), only what a
lidar tracking pipeline needs and skips the ~1 GB of camera images per log:

    sensors/lidar/*.feather       lidar sweeps (~151 MB per log)
    annotations.feather           3D cuboid labels with track ids
    city_SE3_egovehicle.feather   ego poses
    calibration/                  sensor extrinsics/intrinsics
    map/                          vector map + ground-height raster

Source: public S3 bucket, anonymous HTTPS, no login.
Licence of the data: CC BY-NC-SA 4.0 (non-commercial use; do not redistribute).

Every file is checked against the size S3 lists for it; a file that already
exists with the right size is skipped, so the script can be re-run to resume.
A manifest of the chosen logs and every file (key, size, S3 ETag) is written
to <out>/manifest.json.

Usage: python3 fetch_av2.py --out data/av2 --n-logs 20 [--dry-run]
"""
import argparse
import json
import re
import sys
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

BUCKET = "https://argoverse.s3.us-east-1.amazonaws.com"
PREFIX = "datasets/av2/sensor/val/"
KEEP_TOP = ("sensors/lidar/", "annotations.feather", "city_SE3_egovehicle.feather",
            "calibration/", "map/")


def list_objects(prefix, delimiter=None):
    """Yield (key, size, etag) or common prefixes from an S3 ListObjectsV2 listing."""
    token = None
    while True:
        q = {"list-type": "2", "prefix": prefix, "max-keys": "1000"}
        if delimiter:
            q["delimiter"] = delimiter
        if token:
            q["continuation-token"] = token
        xml = urllib.request.urlopen(f"{BUCKET}/?{urllib.parse.urlencode(q)}", timeout=60).read().decode()
        if delimiter:
            yield from re.findall(r"<Prefix>([^<]+)</Prefix>", xml)[1:]  # first is the query prefix
        else:
            for blk in re.findall(r"<Contents>(.*?)</Contents>", xml, re.S):
                key = re.search(r"<Key>([^<]+)</Key>", blk).group(1)
                size = int(re.search(r"<Size>(\d+)</Size>", blk).group(1))
                etag = re.search(r"<ETag>([^<]+)</ETag>", blk).group(1).replace("&quot;", "").strip('"')
                yield key, size, etag
        m = re.search(r"<NextContinuationToken>([^<]+)</NextContinuationToken>", xml)
        if not m:
            return
        token = m.group(1)


def fetch(key, size, dest):
    """Download one object unless it already exists with the expected size."""
    if dest.exists() and dest.stat().st_size == size:
        return "skip"
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    url = f"{BUCKET}/{urllib.parse.quote(key)}"
    for attempt in range(4):
        try:
            with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as f:
                while chunk := r.read(1 << 20):
                    f.write(chunk)
            if tmp.stat().st_size != size:
                raise IOError(f"size {tmp.stat().st_size} != listed {size}")
            tmp.replace(dest)
            return "ok"
        except Exception as e:  # network hiccup: retry
            err = e
    raise RuntimeError(f"{key}: {err}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-logs", type=int, default=20)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    logs = sorted(list_objects(PREFIX, delimiter="/"))
    print(f"{len(logs)} val logs on S3; taking the first {a.n_logs} in sorted order")
    chosen = logs[: a.n_logs]

    files = []
    for lp in chosen:
        for key, size, etag in list_objects(lp):
            rel = key[len(lp):]
            if rel.startswith(KEEP_TOP):
                files.append((key, size, etag))
    total = sum(s for _, s, _ in files)
    print(f"{len(files)} files, {total / 1e9:.2f} GB")
    if a.dry_run:
        return

    a.out.mkdir(parents=True, exist_ok=True)
    manifest = {"source": BUCKET + "/" + PREFIX, "licence": "CC BY-NC-SA 4.0",
                "logs": [lp[len(PREFIX):].rstrip("/") for lp in chosen],
                "files": [{"key": k[len(PREFIX):], "size": s, "etag": e} for k, s, e in files]}
    (a.out / "manifest.json").write_text(json.dumps(manifest, indent=1))

    done = bytes_done = 0
    with ThreadPoolExecutor(a.workers) as ex:
        futs = {ex.submit(fetch, k, s, a.out / "val" / k[len(PREFIX):]): (k, s) for k, s, _ in files}
        for f in as_completed(futs):
            f.result()
            done += 1
            bytes_done += futs[f][1]
            if done % 200 == 0 or done == len(files):
                print(f"  {done}/{len(files)} files, {bytes_done / 1e9:.2f}/{total / 1e9:.2f} GB", flush=True)
    print("all files present with listed sizes")


if __name__ == "__main__":
    sys.exit(main())
