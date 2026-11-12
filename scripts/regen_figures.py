#!/usr/bin/env python
"""Regenerate report-backed figures from a saved ``report.json``.

Figures 2, 3, 6, and 7 are pure functions of the aggregated report, so they can
be reproduced without rerunning training. The latent-space figures (1, 4, 5)
need per-example model outputs and are only produced during a full run.

Usage:
    python scripts/regen_figures.py results/report.json
    python scripts/regen_figures.py results/report.json --output-dir results
"""

from __future__ import annotations

import argparse
import json
import os

from pidjepa.figures import make_report_plots


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "report", nargs="?", default="results/report.json", help="path to report.json"
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="output directory (default: the report's directory)",
    )
    args = parser.parse_args()

    with open(args.report, encoding="utf-8") as handle:
        payload = json.load(handle)
    summary = payload.get("report", payload)
    output_dir = args.output_dir or os.path.dirname(os.path.abspath(args.report))

    print(f"[regen] {args.report} -> {output_dir}")
    make_report_plots(summary, output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
