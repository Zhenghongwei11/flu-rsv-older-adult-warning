from __future__ import annotations

import argparse
import fnmatch
import hashlib
import os
import shutil
import sys
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator, List, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]


DEFAULT_INCLUDE: List[str] = [
    "README.md",
    "LICENSE",
    "CITATION.cff",
    "requirements.txt",
    "requirements-colab.txt",
    "scripts/",
    "tools/",
    "data/manifest.tsv",
    "data/snapshots/",
    "results/",
    "results_strata/",
    "plots/publication/",
    "plots/publication_strata/",
    "notebooks/",
]


DEFAULT_EXCLUDE_GLOBS: List[str] = [
    "**/.DS_Store",
    "**/__pycache__/**",
    "**/*.pyc",
    "**/.ipynb_checkpoints/**",
    ".git/**",
    ".venv/**",
    "venv/**",
    "token",
    # Internal scaffolding (must not be in public bundle)
    "openspec/**",
    "conductor/**",
    # Manuscript / planning docs (keep out of public bundle)
    "docs/**",
    # Large / internal logs
    "logs/openai/**",
    # Legacy backups
    "data/legacy_backups/**",
    # Docx conversion tooling/templates (OpenSpec pipeline requirement)
    "scripts/build_manuscript_docx.py",
    "**/*.docx",
    # Bundle outputs (avoid nesting bundle in itself)
    "docs/review_bundle/**",
    # Submission-only directory
    "submission/**",
]


FIXED_ZIP_DT = (1980, 1, 1, 0, 0, 0)


@dataclass(frozen=True)
class BundleFile:
    rel_path: str
    abs_path: Path
    sha256: str
    size_bytes: int


def _norm_rel(p: str) -> str:
    return p.replace(os.sep, "/").lstrip("./")


def _matches_any(rel_path: str, globs: Sequence[str]) -> bool:
    rp = _norm_rel(rel_path)
    return any(fnmatch.fnmatch(rp, g) for g in globs)


def _sha256_path(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _iter_files(include: Sequence[str], exclude_globs: Sequence[str]) -> Iterator[Path]:
    for item in include:
        item = str(item).strip()
        if not item:
            continue
        abs_item = (REPO_ROOT / item).resolve()
        if not abs_item.exists():
            continue

        if abs_item.is_file():
            rel = _norm_rel(str(abs_item.relative_to(REPO_ROOT)))
            if not _matches_any(rel, exclude_globs):
                yield abs_item
            continue

        # Directory
        for dirpath, dirnames, filenames in os.walk(abs_item):
            dirpath_p = Path(dirpath)
            rel_dir = _norm_rel(str(dirpath_p.relative_to(REPO_ROOT)))
            # Prune excluded directories early
            pruned: List[str] = []
            for dn in list(dirnames):
                rel_dn = _norm_rel(str(Path(rel_dir) / dn)) + "/"
                if _matches_any(rel_dn, exclude_globs):
                    continue
                pruned.append(dn)
            dirnames[:] = pruned

            for fn in filenames:
                abs_f = dirpath_p / fn
                rel_f = _norm_rel(str(abs_f.relative_to(REPO_ROOT)))
                if _matches_any(rel_f, exclude_globs):
                    continue
                yield abs_f


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _sanitize_log_text(text: str) -> str:
    # Remove absolute local paths to avoid leaking personal machines.
    root_str = str(REPO_ROOT).replace("\\", "/")
    out = (text or "").replace(root_str + "/", "")
    out = out.replace(root_str, "")
    return out


def build_bundle(
    outdir: Path,
    bundle_name: str,
    include: Sequence[str],
    exclude_globs: Sequence[str],
    include_logs: bool,
    keep_outdir_clean: bool,
) -> Tuple[Path, List[BundleFile]]:
    outdir = outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    if keep_outdir_clean:
        for p in outdir.iterdir():
            if p.is_file():
                p.unlink()
            else:
                shutil.rmtree(p)

    bundle_zip = outdir / bundle_name
    manifest_path = outdir / "MANIFEST.sha256"
    file_list_path = outdir / "FILE_LIST.txt"
    policy_path = outdir / "INCLUDE_EXCLUDE_POLICY.md"
    reviewer_guide_path = outdir / "REVIEWER_GUIDE.md"

    # Gather files
    paths = sorted({p.resolve() for p in _iter_files(include, exclude_globs)})
    files: List[BundleFile] = []

    staging = outdir / "_staging"
    staging.mkdir(parents=True, exist_ok=True)

    for abs_path in paths:
        rel = _norm_rel(str(abs_path.relative_to(REPO_ROOT)))
        if not include_logs and rel.startswith("logs/"):
            continue

        if rel.startswith("logs/") and abs_path.suffix.lower() == ".log":
            # sanitize log paths
            text = abs_path.read_text(encoding="utf-8", errors="replace")
            sanitized = _sanitize_log_text(text)
            staged_path = staging / rel
            staged_path.parent.mkdir(parents=True, exist_ok=True)
            staged_path.write_text(sanitized, encoding="utf-8")
            abs_for_hash = staged_path
        else:
            # copy file to staging (keeps zip deterministic and avoids symlink surprises)
            staged_path = staging / rel
            staged_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(abs_path, staged_path)
            abs_for_hash = staged_path

        sha = _sha256_path(abs_for_hash)
        files.append(BundleFile(rel_path=rel, abs_path=abs_path, sha256=sha, size_bytes=abs_for_hash.stat().st_size))

    # Write policy + reviewer guide (also included in zip)
    _write_text(
        policy_path,
        "\n".join(
            [
                "# Public review bundle include/exclude policy",
                "",
                "This bundle is intended to be **the canonical public reproducibility package** used for:",
                "- Journal supplementary material (zip upload), and",
                "- GitHub Release assets (same zip; byte-identical).",
                "",
                "## Included (high-level)",
                "- Reproducibility entrypoints: `README.md`, `scripts/run_all.sh`",
                "- Environment lock: `requirements.txt`",
                "- Frozen inputs: `data/snapshots/` + retrieval log `results/dataset_retrieval_log.tsv`",
                "- Source-of-truth results tables: `results/`, `results_strata/`",
                "- Publication figures: `plots/publication/`",
                "",
                "## Excluded (high-level)",
                "- Secrets/credentials: `token`",
                "- Internal scaffolding: `openspec/`, `conductor/`",
                "- Internal agent/tool logs: `logs/openai/`",
                "- Legacy backups: `data/legacy_backups/`",
                "- DOCX conversion tooling/templates and Word files: `scripts/build_manuscript_docx.py`, `*.docx`",
                "- Manuscript / planning docs: `docs/`",
                "- Submission-only artifacts: `submission/`",
                "",
                "## How to verify integrity",
                "- See `MANIFEST.sha256` for per-file SHA-256 checksums.",
                "- See `FILE_LIST.txt` for an ordered inventory of files contained in the zip.",
                "",
            ]
        )
        + "\n",
    )

    _write_text(
        reviewer_guide_path,
        "\n".join(
            [
                "# Reviewer guide (reproducibility)",
                "",
                "## Quick start (recommended: frozen rerun using included snapshots)",
                "```bash",
                "python3 -m venv .venv",
                "source .venv/bin/activate",
                "python3 -m pip install -U pip",
                "python3 -m pip install -r requirements.txt",
                "",
                "SKIP_FETCH=1 ./scripts/run_all.sh",
                "```",
                "",
                "## Expected primary outputs",
                "- `results/analysis/analysis_table.tsv`",
                "- `results/benchmarks/forecast_eval.tsv`",
                "- `results/benchmarks/method_benchmark.tsv`",
                "- `results/benchmarks/paired_benchmark.tsv`",
                "- `plots/publication/`",
                "",
                "## Notes",
                "- The bundle excludes manuscript/submission materials and Word (`.docx`) files by policy.",
                "",
            ]
        )
        + "\n",
    )

    # Stage policy + reviewer guide into zip root for easy discovery.
    for gen, rel_in_zip in [(policy_path, "INCLUDE_EXCLUDE_POLICY.md"), (reviewer_guide_path, "REVIEWER_GUIDE.md")]:
        staged_path = staging / rel_in_zip
        shutil.copy2(gen, staged_path)
        sha = _sha256_path(staged_path)
        files.append(BundleFile(rel_path=rel_in_zip, abs_path=gen, sha256=sha, size_bytes=staged_path.stat().st_size))

    # FILE_LIST should include everything that will appear in the zip, including itself and MANIFEST.
    inventory_paths = sorted(set([bf.rel_path for bf in files] + ["FILE_LIST.txt", "MANIFEST.sha256"]))
    _write_text(file_list_path, "\n".join(inventory_paths) + "\n")
    staged_file_list = staging / "FILE_LIST.txt"
    shutil.copy2(file_list_path, staged_file_list)
    file_list_sha = _sha256_path(staged_file_list)
    files.append(BundleFile(rel_path="FILE_LIST.txt", abs_path=file_list_path, sha256=file_list_sha, size_bytes=staged_file_list.stat().st_size))

    # Manifest excludes itself to make verification straightforward.
    manifest_lines = [f"{bf.sha256}  {bf.rel_path}" for bf in sorted(files, key=lambda x: x.rel_path) if bf.rel_path != "MANIFEST.sha256"]
    _write_text(manifest_path, "\n".join(manifest_lines) + "\n")
    staged_manifest = staging / "MANIFEST.sha256"
    shutil.copy2(manifest_path, staged_manifest)
    manifest_sha = _sha256_path(staged_manifest)
    files.append(BundleFile(rel_path="MANIFEST.sha256", abs_path=manifest_path, sha256=manifest_sha, size_bytes=staged_manifest.stat().st_size))

    # Build deterministic zip from staging
    if bundle_zip.exists():
        bundle_zip.unlink()

    staged_files = sorted([p for p in staging.rglob("*") if p.is_file()], key=lambda p: _norm_rel(str(p.relative_to(staging))))
    with zipfile.ZipFile(bundle_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for p in staged_files:
            rel_in_zip = _norm_rel(str(p.relative_to(staging)))
            zi = zipfile.ZipInfo(filename=rel_in_zip, date_time=FIXED_ZIP_DT)
            zi.compress_type = zipfile.ZIP_DEFLATED
            zi.external_attr = 0o644 << 16
            with p.open("rb") as f:
                zf.writestr(zi, f.read())

    # Cleanup staging directory
    shutil.rmtree(staging)

    return bundle_zip, sorted(files, key=lambda x: x.rel_path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build a sanitized public review bundle zip for GitHub release / journal supplement.")
    parser.add_argument("--outdir", default="release/review_bundle", help="Output directory (default: release/review_bundle).")
    parser.add_argument("--name", default="", help="Bundle zip filename (default: auto).")
    parser.add_argument("--no-logs", action="store_true", help="Exclude logs/ entirely.")
    parser.add_argument("--keep", action="store_true", help="Do not clean outdir before building.")
    args = parser.parse_args(list(argv) if argv is not None else None)

    outdir = (REPO_ROOT / args.outdir).resolve() if not os.path.isabs(args.outdir) else Path(args.outdir).resolve()

    if args.name:
        name = args.name
    else:
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        name = f"public_review_bundle_{ts}.zip"

    bundle_zip, files = build_bundle(
        outdir=outdir,
        bundle_name=name,
        include=DEFAULT_INCLUDE,
        exclude_globs=DEFAULT_EXCLUDE_GLOBS,
        include_logs=not args.no_logs,
        keep_outdir_clean=not args.keep,
    )

    zip_sha = _sha256_path(bundle_zip)
    print(f"Built: {bundle_zip.relative_to(REPO_ROOT)}")
    print(f"SHA256: {zip_sha}")
    print(f"Files: {len(files)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
