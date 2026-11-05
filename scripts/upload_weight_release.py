"""Upload a staged checkpoint release and verify its remote file hashes."""

import argparse
import json
import os
import re
from pathlib import Path

os.environ.setdefault("HF_XET_CACHE", str(Path(".cache/hf_xet").resolve()))
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

from huggingface_hub import HfApi


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=Path)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--source-commit")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    manifest_path = args.folder / "CHECKPOINTS.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest["model_id"] != args.repo_id or manifest["checkpoint_count"] != 30:
        raise ValueError("Unexpected release destination or checkpoint count")
    if not 1 <= args.workers <= 4:
        raise ValueError("Use one to four upload workers")
    if args.source_commit:
        if not re.fullmatch(r"[0-9a-f]{40}", args.source_commit):
            raise ValueError("Use a full Git commit hash")
        manifest["source_revision"] = args.source_commit
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        card_path = args.folder / "README.md"
        card = card_path.read_text()
        marker = "## Provenance\n"
        link = f"Code snapshot: [{args.source_commit}]({manifest['github']}/tree/{args.source_commit}).\n\n"
        if link not in card:
            card_path.write_text(card.replace(marker, link + marker, 1))
    for row in manifest["checkpoints"]:
        path = args.folder / row["file"]
        if not path.is_file() or path.stat().st_size != row["bytes"]:
            raise ValueError(f"Missing or truncated checkpoint: {row['file']}")
    api = HfApi()
    api.upload_large_folder(
        repo_id=args.repo_id, folder_path=args.folder, repo_type="model",
        num_workers=args.workers, print_report=True, print_report_every=60,
        allow_patterns=["*.ckpt", "README.md", "LICENSE", "CHECKPOINTS.json", "SHA256SUMS",
                        "edits/*.json", "evaluation/*.md", "evaluation/*.json"],
        ignore_patterns=[".cache/**"],
    )
    info = api.model_info(args.repo_id, files_metadata=True)
    remote = {file.rfilename: file for file in info.siblings}
    verified = []
    for row in manifest["checkpoints"]:
        file = remote.get(row["file"])
        if file is None or file.size != row["bytes"] or file.lfs is None:
            raise ValueError(f"Remote checkpoint metadata missing: {row['file']}")
        if file.lfs.sha256 != row["sha256"]:
            raise ValueError(f"Remote checkpoint hash mismatch: {row['file']}")
        verified.append(row["file"])
    expected = {row["file"] for row in manifest["checkpoints"]}
    if {name for name in remote if name.endswith(".ckpt")} != expected:
        raise ValueError("Unexpected remote checkpoint files")
    result = {
        "repo_id": args.repo_id, "revision": info.sha, "private": info.private,
        "verified_checkpoints": verified, "all_checkpoint_hashes_match": True,
        "logical_bytes": manifest["total_logical_bytes"],
    }
    (args.folder / "upload_verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
