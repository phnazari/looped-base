"""Tokenize one source of the mix into OLMo-core token shards, as Dolma's `dolma tokens` does for OLMo 2.

    tokenize_source.py "INPUT_GLOB" OUTPUT_DIR --tokenizer-dir DIR [--text-field text] [--workers 40]
        [--ring-size 8] [--batch-size 10000] [--max-size 1073741824] [--seed 0]

The input parquet files are shuffled (seeded) and dealt to the workers. Each worker keeps RING_SIZE files
open, fills a buffer of BATCH_SIZE documents by reading from them in turn, shuffles it, tokenizes it with
the dolma2 tokenizer and appends it to its current shard: a flat uint32 token array (part-WWW-SSSSS.npy,
raw like OLMo-core's numpy shards) with EOS (100257) after every document and no BOS; special-token
strings inside documents are tokenized as plain text. A shard is closed at MAX_SIZE bytes. Next to each
shard: part-WWW-SSSSS.csv.gz (start,end,source file,row of every document) and part-WWW-SSSSS.meta.json
(documents, tokens). A worker that finished writes worker-WWW.done and is skipped on a rerun; an
unfinished worker's shards are removed and redone. --tokenizer-dir is a local copy of the dolma2 tokenizer
(AutoTokenizer.from_pretrained("allenai/dolma2-tokenizer").save_pretrained(DIR)): loaded by name, the
workers would need the Hub, and offline the name fails on the repo's missing config.json.
"""
import argparse
import csv
import gzip
import json
import multiprocessing
import os
import random
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

TOKENIZER = "allenai/dolma2-tokenizer"
EOS = 100257


class Shards:
    def __init__(self, output_dir, worker, max_size):
        self.output_dir, self.worker, self.max_size = output_dir, worker, max_size
        self.index = -1
        self.open_next()

    def open_next(self):
        self.index += 1
        self.stem = self.output_dir / f"part-{self.worker:03d}-{self.index:05d}"
        self.data = open(f"{self.stem}.npy.tmp", "wb")
        self.rows = gzip.open(f"{self.stem}.csv.gz.tmp", "wt", compresslevel=6)
        self.writer = csv.writer(self.rows)
        self.tokens = self.documents = 0

    def write(self, ids, source, row):
        array = np.asarray(ids + [EOS], dtype=np.uint32)
        array.tofile(self.data)
        self.writer.writerow([self.tokens, self.tokens + len(array), source, row])
        self.tokens += len(array)
        self.documents += 1
        if self.tokens * 4 >= self.max_size:
            self.close()
            self.open_next()

    def close(self):
        self.data.close()
        self.rows.close()
        if self.documents == 0:
            os.remove(f"{self.stem}.npy.tmp")
            os.remove(f"{self.stem}.csv.gz.tmp")
            return
        os.replace(f"{self.stem}.npy.tmp", f"{self.stem}.npy")
        os.replace(f"{self.stem}.csv.gz.tmp", f"{self.stem}.csv.gz")
        meta = {"documents": self.documents, "tokens": self.tokens, "dtype": "uint32", "eos_token_id": EOS,
                "bos_token_id": None, "tokenizer": TOKENIZER}
        Path(f"{self.stem}.meta.json").write_text(json.dumps(meta) + "\n")


def batches(path, text_field):
    for batch in pq.ParquetFile(path).iter_batches(batch_size=1000, columns=[text_field]):
        yield batch.column(0).to_pylist()


def run_worker(job):
    worker, files, args = job
    output_dir = Path(args.output_dir)
    done = output_dir / f"worker-{worker:03d}.done"
    if done.exists():
        return json.loads(done.read_text())
    for stale in output_dir.glob(f"part-{worker:03d}-*"):
        stale.unlink()
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir)
    rng = random.Random(args.seed * 1000 + worker)
    shards = Shards(output_dir, worker, args.max_size)
    pending = list(files)
    ring = []  # [path, iterator, rows read]
    buffer = []
    totals = {"documents": 0, "tokens": 0, "files": len(files)}

    def flush():
        rng.shuffle(buffer)
        encoded = tokenizer([text for text, _, _ in buffer], add_special_tokens=False, split_special_tokens=True,
                            return_attention_mask=False)["input_ids"]
        for ids, (_, source, row) in zip(encoded, buffer):
            shards.write(ids, source, row)
            totals["documents"] += 1
            totals["tokens"] += len(ids) + 1
        buffer.clear()

    while pending or ring:
        while pending and len(ring) < args.ring_size:
            path = pending.pop(0)
            ring.append([path, batches(path, args.text_field), 0])
        for entry in list(ring):
            path, iterator, offset = entry
            texts = next(iterator, None)
            if texts is None:
                ring.remove(entry)
                continue
            buffer.extend((text or "", Path(path).name, offset + i) for i, text in enumerate(texts))
            entry[2] = offset + len(texts)
            if len(buffer) >= args.batch_size:
                flush()
    if buffer:
        flush()
    shards.close()
    done.write_text(json.dumps(totals) + "\n")
    return totals


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input_glob")
    parser.add_argument("output_dir")
    parser.add_argument("--tokenizer-dir", required=True)
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--workers", type=int, default=40)
    parser.add_argument("--ring-size", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=10000)
    parser.add_argument("--max-size", type=int, default=1 << 30)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    from glob import glob

    files = sorted(glob(args.input_glob))
    if not files:
        raise SystemExit(f"no files match {args.input_glob}")
    random.Random(args.seed).shuffle(files)
    workers = min(args.workers, len(files))
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    jobs = [(w, files[w::workers], args) for w in range(workers)]
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    totals = {"documents": 0, "tokens": 0, "files": 0}
    with multiprocessing.get_context("spawn").Pool(workers) as pool:
        for result in pool.imap_unordered(run_worker, jobs):
            for key in totals:
                totals[key] += result[key]
            print(json.dumps({"worker_done": result, "total": totals}), flush=True)
    summary = {"input_glob": args.input_glob, "files": len(files), "workers": workers, "seed": args.seed,
               "tokenizer": TOKENIZER, **totals}
    (Path(args.output_dir) / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print("done", json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
