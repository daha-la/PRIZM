#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from datetime import datetime, timezone
import socket

def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def write_done(done_path: Path, phase: str, msa_id: str, evidence: str, overwrite: bool) -> bool:
    """
    Returns True if written, False if skipped due to existing file.
    """
    if done_path.exists() and not overwrite:
        return False

    done_path.parent.mkdir(parents=True, exist_ok=True)

    tmp = done_path.with_suffix(done_path.suffix + ".tmp")
    content = (
        f"ok {now_iso()}\n"
        f"phase={phase}\n"
        f"msa_id={msa_id}\n"
        f"evidence={evidence}\n"
        f"host={socket.gethostname()}\n"
        f"backfilled=true\n"
    )
    tmp.write_text(content)
    tmp.replace(done_path)
    return True

def strip_after_dot(name: str) -> str:
    # match your current MSA identity policy: strip everything after first dot
    return name.split(".", 1)[0]

def find_eve_msa_ids(eve_dir: Path) -> dict[str, str]:
    """
    Infer msa_id from EVE checkpoint filenames like:
      A4_HUMAN_2023-08-07_b01_seed_42
      A4_HUMAN_2023-08-07_b01_seed_42_step_0
    Return dict msa_id -> evidence string
    """
    # accept: <msa_id>_seed_<digits>(optional _step_<digits>)
    pat = re.compile(r"^(?P<msa>.+)_seed_\d+(?:_step_\d+)?$")

    msa_to_evidence: dict[str, str] = {}

    if not eve_dir.exists():
        return msa_to_evidence

    for p in eve_dir.iterdir():
        if not p.is_file():
            continue
        if p.name in {"README.md", "log_prior"}:
            continue

        m = pat.match(p.name)
        if not m:
            continue

        msa_id = strip_after_dot(m.group("msa"))
        # prefer "main" checkpoint evidence over step_0 if both exist
        # we can just keep the shortest matching evidence name.
        evidence = f"file:{p.name}"
        if msa_id not in msa_to_evidence:
            msa_to_evidence[msa_id] = evidence
        else:
            # keep the "better" evidence (heuristic: shorter filename usually main checkpoint)
            if len(p.name) < len(msa_to_evidence[msa_id].split("file:", 1)[1]):
                msa_to_evidence[msa_id] = evidence

    return msa_to_evidence

def find_eunirep_msa_ids(eunirep_dir: Path) -> dict[str, str]:
    """
    Infer msa_id from eUniRep folder names.
    Example:
      finetuned_models/eUniRep/A4_HUMAN_2023-08-07_b01/  (contains .npy or _1k/_2k etc)
    Return dict msa_id -> evidence string
    """
    msa_to_evidence: dict[str, str] = {}

    if not eunirep_dir.exists():
        return msa_to_evidence

    for p in eunirep_dir.iterdir():
        if not p.is_dir():
            continue
        if p.name in {"README.md", "done"}:
            continue

        msa_id = strip_after_dot(p.name)

        # evidence: check for any .npy or *_k subfolders
        npys = list(p.glob("*.npy"))
        kdirs = [d for d in p.iterdir() if d.is_dir() and re.match(r"^_\d+k$", d.name)]

        if npys:
            msa_to_evidence[msa_id] = f"dir:{p.name} (npy:{npys[0].name})"
        elif kdirs:
            msa_to_evidence[msa_id] = f"dir:{p.name} (subdir:{kdirs[0].name})"
        else:
            # still counts as "exists", but evidence is weaker
            msa_to_evidence[msa_id] = f"dir:{p.name} (empty_or_unknown)"

    return msa_to_evidence

def main():
    ap = argparse.ArgumentParser(description="Backfill PRIZM DONE markers from existing EVE/eUniRep artifacts.")
    ap.add_argument("--prizm-path", required=True, help="Path to PRIZM root (contains finetuned_models/)")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing DONE files")
    ap.add_argument("--dry-run", action="store_true", help="Only report what would be written")
    args = ap.parse_args()

    prizm = Path(args.prizm_path).expanduser().resolve()
    finetuned = prizm / "finetuned_models"

    eve_dir = finetuned / "EVE"
    eunirep_dir = finetuned / "eUniRep"

    eve_done_dir = eve_dir / "done"
    eunirep_done_dir = eunirep_dir / "done"

    eve_msa = find_eve_msa_ids(eve_dir)
    eunirep_msa = find_eunirep_msa_ids(eunirep_dir)

    print(f"PRIZM: {prizm}")
    print(f"EVE dir: {eve_dir} (found {len(eve_msa)} model ids)")
    print(f"eUniRep dir: {eunirep_dir} (found {len(eunirep_msa)} model ids)")
    print()

    def backfill(phase: str, msa_map: dict[str, str], done_dir: Path):
        wrote = 0
        skipped = 0
        for msa_id in sorted(msa_map.keys()):
            done_path = done_dir / f"{msa_id}.txt"
            evidence = msa_map[msa_id]
            if args.dry_run:
                status = "WOULD_WRITE" if (args.overwrite or not done_path.exists()) else "SKIP_EXISTS"
                print(f"[{phase}] {status} {done_path}  ({evidence})")
                continue

            ok = write_done(done_path, phase=phase, msa_id=msa_id, evidence=evidence, overwrite=args.overwrite)
            if ok:
                wrote += 1
                print(f"[{phase}] wrote {done_path}  ({evidence})")
            else:
                skipped += 1
                print(f"[{phase}] skip (exists) {done_path}")

        if not args.dry_run:
            print(f"\n[{phase}] summary: wrote={wrote}, skipped_existing={skipped}\n")

    backfill("EVE", eve_msa, eve_done_dir)
    backfill("eUniRep", eunirep_msa, eunirep_done_dir)

if __name__ == "__main__":
    main()
