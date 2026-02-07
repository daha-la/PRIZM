#!/usr/bin/env python3
import argparse
import csv
import sys
from collections import OrderedDict


def parse_args():
    p = argparse.ArgumentParser(
        description="Extract unique MSA_filename values and representative indices from a PRIZM reference CSV."
    )
    p.add_argument("--ref", required=True, help="Path to reference CSV")
    p.add_argument("--first", type=int, required=True, help="First index (0-based, inclusive)")
    p.add_argument("--last", type=int, required=True, help="Last index (0-based, inclusive)")
    p.add_argument("--col", default="MSA_filename", help="Column name for MSA filename (default: MSA_filename)")
    p.add_argument("--header", action="store_true", help="Print a header row")
    return p.parse_args()


def main():
    args = parse_args()
    if args.first < 0 or args.last < 0:
        print("ERROR: --first/--last must be >= 0", file=sys.stderr)
        return 2
    if args.first > args.last:
        print("ERROR: --first must be <= --last", file=sys.stderr)
        return 2

    msa_to_rep = OrderedDict()  # preserves first-seen order

    with open(args.ref, "r", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            print("ERROR: Reference file has no header row.", file=sys.stderr)
            return 2
        if args.col not in reader.fieldnames:
            print(f"ERROR: Column '{args.col}' not found in reference CSV header.", file=sys.stderr)
            print(f"Header columns: {reader.fieldnames}", file=sys.stderr)
            return 2

        for idx, row in enumerate(reader):
            if idx < args.first:
                continue
            if idx > args.last:
                break

            msa = (row.get(args.col) or "").strip()
            if not msa:
                # If a row in-range lacks an MSA filename, fail fast
                print(f"ERROR: Empty '{args.col}' at row index {idx}", file=sys.stderr)
                return 2

            if msa not in msa_to_rep:
                msa_to_rep[msa] = idx  # representative index = first occurrence in range

    if not msa_to_rep:
        print("ERROR: No rows found in the given index range.", file=sys.stderr)
        return 2

    out = sys.stdout
    if args.header:
        out.write("MSA_filename\trep_index\n")
    for msa, rep_idx in msa_to_rep.items():
        out.write(f"{msa}\t{rep_idx}\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
