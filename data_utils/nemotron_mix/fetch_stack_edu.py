"""Fetch the code files of a random subset of Stack-Edu from Software Heritage's S3 bucket.

    fetch_stack_edu.py INDEX_DIR OUTPUT_DIR --fraction F [--threads 256] [--seed 0] [--part I --parts N]

INDEX_DIR holds the Hugging Face index (HuggingFaceTB/stack-edu: <Language>/train-*.parquet with the
SWH blob ids). A file is selected when a seeded hash of its blob id falls below FRACTION, so the subset
is the same on every rerun, spread evenly over languages and index files, and a larger fraction only
adds files. Each index file becomes one parquet of fetched files under OUTPUT_DIR/<Language>/ (columns
text, language, repo_name, path, blob_id, license_type, int_score), written atomically; an index file
whose output exists is skipped, so an interrupted run resumes. Blobs are gzip-compressed on S3.
--part I --parts N takes every N-th index file from the I-th: N processes fetch disjoint index files.
"""
import argparse
import gzip
import hashlib
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import requests
from requests.adapters import HTTPAdapter

URL = "https://softwareheritage.s3.amazonaws.com/content/{}"
COLUMNS = ["blob_id", "language", "repo_name", "path", "src_encoding", "license_type", "int_score"]
local = threading.local()


def session(threads):
    if not hasattr(local, "session"):
        local.session = requests.Session()
        local.session.mount("https://", HTTPAdapter(pool_connections=4, pool_maxsize=threads))
    return local.session


def selected(blob_id, seed, fraction):
    digest = hashlib.blake2b(f"{seed}:{blob_id}".encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big") < fraction * 2**64


def fetch(row, threads):
    for attempt in range(4):
        try:
            response = session(threads).get(URL.format(row["blob_id"]), timeout=60)
            if response.status_code == 200:
                text = gzip.decompress(response.content).decode(row["src_encoding"] or "utf-8", errors="replace")
                return {"text": text, **{k: row[k] for k in COLUMNS if k != "src_encoding"}}
            if response.status_code == 404:
                return None
        except (requests.RequestException, OSError, LookupError):
            pass
        time.sleep(2 ** attempt)
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("index_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--fraction", type=float, required=True)
    parser.add_argument("--threads", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit-index-files", type=int)
    parser.add_argument("--part", type=int, default=0)
    parser.add_argument("--parts", type=int, default=1)
    args = parser.parse_args()
    index_files = sorted(args.index_dir.glob("*/*.parquet"))[: args.limit_index_files][args.part :: args.parts]
    totals = {"selected": 0, "fetched": 0, "bytes": 0}
    with ThreadPoolExecutor(args.threads) as pool:
        for index_file in index_files:
            language = index_file.parent.name
            output = args.output_dir / language / index_file.name
            if output.exists():
                continue
            table = pq.read_table(index_file, columns=COLUMNS)
            rows = [row for row in table.to_pylist() if selected(row["blob_id"], args.seed, args.fraction)]
            begin = time.perf_counter()
            files = [f for f in pool.map(lambda row: fetch(row, args.threads), rows) if f is not None]
            output.parent.mkdir(parents=True, exist_ok=True)
            temporary = output.with_suffix(".parquet.tmp")
            pq.write_table(pa.Table.from_pylist(files), temporary, compression="zstd")
            temporary.rename(output)
            size = sum(len(f["text"]) for f in files)
            totals["selected"] += len(rows)
            totals["fetched"] += len(files)
            totals["bytes"] += size
            print(json.dumps({"index_file": f"{language}/{index_file.name}", "selected": len(rows),
                              "fetched": len(files), "chars": size,
                              "files_per_s": round(len(rows) / (time.perf_counter() - begin), 1),
                              **{f"total_{k}": v for k, v in totals.items()}}), flush=True)
    print("done", json.dumps(totals), flush=True)


if __name__ == "__main__":
    sys.exit(main())
