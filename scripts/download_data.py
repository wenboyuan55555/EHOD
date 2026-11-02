#!/usr/bin/env python3
"""Download EHOD data releases and verify archives and extracted file hashes."""
import argparse
import hashlib
import json
import os
import tempfile
import time
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BLOCK = 8 * 1024 * 1024


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(BLOCK), b""):
            digest.update(block)
    return digest.hexdigest()


def matches(path, item):
    return (path.is_file() and path.stat().st_size == item["bytes"]
            and sha256(path) == item["sha256"])


def download_archive(asset, directory):
    directory.mkdir(parents=True, exist_ok=True)
    archive = directory / asset["name"]
    if matches(archive, asset):
        print("Using verified archive: " + str(archive), flush=True)
        return archive
    temporary = archive.with_suffix(archive.suffix + ".part")
    print("Downloading %s (%.1f MiB)" % (asset["name"], asset["bytes"] / 1024 ** 2), flush=True)
    for attempt in range(3):
        try:
            request = urllib.request.Request(asset["url"], headers={"User-Agent": "EHOD-data-downloader"})
            digest = hashlib.sha256()
            received = 0
            last_report = time.monotonic()
            with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as target:
                for block in iter(lambda: response.read(BLOCK), b""):
                    target.write(block)
                    digest.update(block)
                    received += len(block)
                    if received > asset["bytes"]:
                        raise RuntimeError("Download exceeds expected size: " + asset["name"])
                    if time.monotonic() - last_report >= 10:
                        print("  %.1f%%" % (received * 100 / asset["bytes"]), flush=True)
                        last_report = time.monotonic()
            if received != asset["bytes"] or digest.hexdigest() != asset["sha256"]:
                raise RuntimeError("Downloaded archive size/SHA-256 mismatch: " + asset["name"])
            temporary.replace(archive)
            return archive
        except (OSError, RuntimeError) as error:
            if attempt == 2:
                raise
            print("Download failed (%s); retrying..." % error, flush=True)
            time.sleep(2)


def destination(data_root, item):
    relative = Path(item["path"]).relative_to("data")
    target = (data_root / relative).resolve()
    if data_root != target and data_root not in target.parents:
        raise ValueError("Unsafe data path: " + item["path"])
    return target


def install_archive(archive, items, data_root):
    with zipfile.ZipFile(archive) as bundle:
        if sorted(bundle.namelist()) != sorted(item["path"] for item in items):
            raise RuntimeError("ZIP contents do not match data/manifest.json")
        for item in items:
            target = destination(data_root, item)
            if matches(target, item):
                print("Verified existing " + item["path"], flush=True)
                continue
            if bundle.getinfo(item["path"]).file_size != item["bytes"]:
                raise RuntimeError("ZIP member size mismatch: " + item["path"])
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = None
            try:
                digest = hashlib.sha256()
                size = 0
                with tempfile.NamedTemporaryFile(dir=str(target.parent), prefix=".ehod-", delete=False) as output:
                    temporary = Path(output.name)
                    with bundle.open(item["path"]) as source:
                        for block in iter(lambda: source.read(BLOCK), b""):
                            output.write(block)
                            digest.update(block)
                            size += len(block)
                if size != item["bytes"] or digest.hexdigest() != item["sha256"]:
                    raise RuntimeError("Extracted data size/SHA-256 mismatch: " + item["path"])
                temporary.replace(target)
                print("Installed and verified " + item["path"], flush=True)
            finally:
                if temporary is not None and temporary.exists():
                    temporary.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--city", choices=("nyc", "chicago", "all"), default="all")
    parser.add_argument("--data-root", type=Path,
                        default=Path(os.environ.get("EHOD_DATA_ROOT", str(ROOT / "data"))))
    parser.add_argument("--download-dir", type=Path, default=ROOT / ".downloads")
    args = parser.parse_args()
    data_root = args.data_root.resolve()
    release = json.loads((ROOT / "data/releases/data-v1.json").read_text(encoding="utf-8"))
    manifest = json.loads((ROOT / "data/manifest.json").read_text(encoding="utf-8"))
    cities = ("nyc", "chicago") if args.city == "all" else (args.city,)
    for city in cities:
        items = [item for item in manifest["files"] if item["path"].startswith("data/" + city + "/")]
        if all(matches(destination(data_root, item), item) for item in items):
            print("%s: all %d data files already present and verified" % (city, len(items)), flush=True)
            continue
        asset = next(asset for asset in release["assets"] if asset["city"] == city)
        archive = download_archive(asset, args.download_dir)
        install_archive(archive, items, data_root)
        print("%s: all %d data files installed and verified" % (city, len(items)), flush=True)


if __name__ == "__main__":
    main()
