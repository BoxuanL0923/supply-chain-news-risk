"""Prepare an operator-review manifest/patch for an explicit candidate commit.

Read-only git commands. Excluded dirty files use HEAD through checkout filters,
preventing certification of active ablation changes. Clean files retain this
checkout's byte representation (including its existing line-ending choices).
Existing outputs are never overwritten. No .git or repository file is changed.
"""
from __future__ import annotations

import argparse
import difflib
import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess


def prepare(repo, files, output, patch, snapshot=None):
    repo = Path(repo).resolve()
    def git(*args):
        return subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True).stdout
    selected = set()
    for line in Path(files).read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        relative = PurePosixPath(line.strip())
        if relative.is_absolute() or ".." in relative.parts or relative.parts[0] == ".git" or ":" in str(relative):
            raise ValueError("Unsafe commit-list path")
        selected.add(str(relative))
    manifest_path = "metadata/file_manifest.json"
    tracked = {p.decode("utf-8") for p in git("ls-files", "-z").split(b"\0") if p}
    dirty = {p.decode("utf-8") for p in git("diff", "--name-only", "-z", "HEAD").split(b"\0") if p}
    entries, contents = [], {}
    for name in sorted((tracked | selected) - {manifest_path}):
        if name in selected or name not in dirty:
            path = repo / name
            if path.is_symlink() or not path.is_file():
                raise ValueError(f"Missing or symlink candidate: {name}")
            blob = path.read_bytes()
        else:
            blob = git("cat-file", "--filters", "HEAD:" + name)
        contents[name] = blob
        entries.append(dict(path=name, bytes=len(blob), sha256=hashlib.sha256(blob).hexdigest()))
    candidate = json.dumps(dict(hash="sha256", files=entries), indent=2) + "\n"
    old = (repo / manifest_path).read_text(encoding="utf-8")
    diff = "".join(difflib.unified_diff(old.splitlines(keepends=True), candidate.splitlines(keepends=True),
                                       fromfile="a/"+manifest_path, tofile="b/"+manifest_path))
    for path, value in ((output, candidate), (patch, diff)):
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(value)
    if snapshot:
        snapshot = Path(snapshot)
        snapshot.mkdir(parents=True, exist_ok=False)
        contents[manifest_path] = candidate.encode()
        for name, blob in contents.items():
            path = snapshot / name
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as stream:
                stream.write(blob)
    return dict(files=len(entries), selected=sorted(selected), manifest=str(output), patch=str(patch),
                snapshot=str(snapshot) if snapshot else None,
                note="Excluded dirty files use HEAD with checkout filters; hashes are checkout-byte-specific")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--files", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--patch", required=True)
    parser.add_argument("--snapshot")
    args = parser.parse_args()
    print(json.dumps(prepare(args.repo, args.files, args.out, args.patch, args.snapshot), indent=2))
