"""Download part of a Hugging Face dataset repo into a local directory (no cache copy).

    download_hf.py REPO_ID LOCAL_DIR PATTERN [PATTERN ...]

Reads HF_TOKEN from the environment for gated repos."""
import os
import sys

from huggingface_hub import snapshot_download

repo_id, local_dir, patterns = sys.argv[1], sys.argv[2], sys.argv[3:]
path = snapshot_download(repo_id=repo_id, repo_type="dataset", local_dir=local_dir, allow_patterns=patterns,
                         token=os.environ.get("HF_TOKEN"), max_workers=8)
print("done", path, flush=True)
