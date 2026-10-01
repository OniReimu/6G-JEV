"""Tree provenance for git-less runs.

On a clean checkout (the Mac), build_provenance() writes PROVENANCE.json at the repo root: the HEAD
commit and the sha256 of every `git ls-files` path plus every file under DATA_TREES (the corpus, tracked
or not). On a node without git, verify_provenance() re-hashes every listed file and refuses on any
mismatch or missing file, and verify_listed_file() binds an input (the --cases file) to that list.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
PROVENANCE_FILE = "PROVENANCE.json"
# Listed from disk whether tracked, untracked or gitignored.
DATA_TREES = ("data/edgebench/v1",)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def tree_sha256(file_hashes: dict[str, str]) -> str:
    """sha256 of the sorted "<sha256>  <path>" listing."""
    listing = "".join(f"{file_hashes[p]}  {p}\n" for p in sorted(file_hashes))
    return hashlib.sha256(listing.encode("utf-8")).hexdigest()


def _git(root: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=root, check=True, capture_output=True, text=True).stdout


def build_provenance(root: Path) -> dict:
    """Write PROVENANCE.json for a clean checkout; refuses when `git status --porcelain` is non-empty
    (untracked files under DATA_TREES excepted)."""
    root = Path(root)
    status = _git(root, "status", "--porcelain", "--untracked-files=all")
    # Untracked corpus files are allowed: they are hashed below like tracked ones.
    dirty = [line for line in status.splitlines() if not (
        line.startswith("?? ") and line[3:].strip('"').startswith(tuple(t + "/" for t in DATA_TREES)))]
    if dirty:
        raise RuntimeError("Refusing to write provenance: working tree is not clean:\n" + "\n".join(dirty))
    head = _git(root, "rev-parse", "HEAD").strip()
    paths = {p for p in _git(root, "ls-files", "-z").split("\0") if p}
    for tree in DATA_TREES:
        if (root / tree).is_dir():
            paths.update(f.relative_to(root).as_posix() for f in (root / tree).rglob("*") if f.is_file())
    files = {p: sha256_file(root / p) for p in sorted(paths)}
    data = {
        "head_sha": head,
        "tree_sha256": tree_sha256(files),
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "files": files,
    }
    with open(root / PROVENANCE_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=1, sort_keys=True)
        f.write("\n")
    return data


def verify_provenance(root: Path | None = None) -> tuple[str, str]:
    """Re-hash every file listed in PROVENANCE.json; returns (head_sha, tree_sha256) or raises RuntimeError."""
    root = Path(root) if root is not None else REPO_ROOT
    path = root / PROVENANCE_FILE
    if not path.is_file():
        raise RuntimeError(
            f"Refusing to start: git is unavailable and {path} is missing. "
            "Run scripts/eb_provenance.py on a clean checkout before syncing."
        )
    data = json.loads(path.read_text(encoding="utf-8"))
    head = str(data.get("head_sha", ""))
    files = data.get("files")
    if not re.fullmatch(r"[0-9a-f]{40}", head) or not isinstance(files, dict) or not files:
        raise RuntimeError(f"Refusing to start: {path} is malformed (head_sha / files).")

    missing = sorted(p for p in files if not (root / p).is_file())
    mismatched = sorted(p for p in files if p not in missing and sha256_file(root / p) != files[p])
    if missing or mismatched:
        raise RuntimeError(
            "Refusing to start: synced tree does not match PROVENANCE.json. "
            f"missing={missing[:10]} ({len(missing)}) mismatched={mismatched[:10]} ({len(mismatched)})"
        )
    tree = tree_sha256(files)
    if data.get("tree_sha256") != tree:
        raise RuntimeError(f"Refusing to start: tree_sha256 in {path} does not match its file list.")
    return head, tree


def verify_listed_file(path: str | Path, root: Path | None = None) -> str:
    """Require `path` (inside the repo) to be listed in PROVENANCE.json with its current sha256; returns it.

    Raises RuntimeError when it lies outside the repo, is not listed, or its hash does not match.
    """
    root = Path(root) if root is not None else REPO_ROOT
    resolved = Path(path).resolve()
    try:
        rel = resolved.relative_to(root.resolve()).as_posix()
    except ValueError:
        raise RuntimeError(f"Refusing to start: {path} is outside the provenance root {root}.") from None
    files = json.loads((root / PROVENANCE_FILE).read_text(encoding="utf-8")).get("files") or {}
    if rel not in files:
        raise RuntimeError(f"Refusing to start: {rel} is not listed in {PROVENANCE_FILE}.")
    digest = sha256_file(resolved)
    if files[rel] != digest:
        raise RuntimeError(f"Refusing to start: sha256 of {rel} does not match {PROVENANCE_FILE}.")
    return digest
