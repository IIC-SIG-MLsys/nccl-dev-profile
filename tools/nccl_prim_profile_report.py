#!/usr/bin/env python3
"""Analyze NCCL primitive profiling CSV outputs and generate summary tables + plots.

Input CSV format is produced by NCCL_PRIM_PROFILE_FILE with header:
  type,channel,work,tb_cycles,prim_cycles_total,prim,cycles,calls,pct_tb,pct_prim_sum,start_clk,stop_clk
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple


@dataclass
class WorkRecord:
    source: str
    channel: int
    work: int
    tb_cycles: int = 0
    prim_cycles_total: int = 0
    start_clk: int = 0
    stop_clk: int = 0


def _to_int(v: str) -> int:
    if v is None:
        return 0
    s = str(v).strip()
    if s == "":
        return 0
    try:
        return int(float(s))
    except ValueError:
        return 0


def _to_float(v: str) -> float:
    if v is None:
        return 0.0
    s = str(v).strip()
    if s == "":
        return 0.0
    try:
        return float(s)
    except ValueError:
        return 0.0


def resolve_inputs(patterns: Iterable[str]) -> List[str]:
    paths: List[str] = []
    for p in patterns:
        matched = sorted(glob.glob(os.path.expanduser(p)))
        if matched:
            paths.extend(matched)
        elif os.path.isfile(p):
            paths.append(p)
    # De-dup while preserving order
    seen = set()
    out = []
    for p in paths:
        if p in seen:
            continue
        seen.add(p)
        out.append(p)
    return out


def load_data(files: List[str]):
    prim_rows: List[dict] = []
    works: Dict[Tuple[str, int, int], WorkRecord] = {}

    for src in files:
        with open(src, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                typ = (row.get("type") or "").strip()
                if typ not in ("tb", "prim"):
                    continue

                channel = _to_int(row.get("channel", "0"))
                work = _to_int(row.get("work", "0"))
                key = (src, channel, work)

                if key not in works:
                    works[key] = WorkRecord(source=src, channel=channel, work=work)

                rec = works[key]
                tb_cycles = _to_int(row.get("tb_cycles", "0"))
                prim_cycles_total = _to_int(row.get("prim_cycles_total", "0"))
                start_clk = _to_int(row.get("start_clk", "0"))
                stop_clk = _to_int(row.get("stop_clk", "0"))

                rec.tb_cycles = max(rec.tb_cycles, tb_cycles)
                rec.prim_cycles_total = max(rec.prim_cycles_total, prim_cycles_total)
                rec.start_clk = rec.start_clk or start_clk
                rec.stop_clk = max(rec.stop_clk, stop_clk)

                if typ == "prim":
                    prim_rows.append(
                        {
                            "source": src,
                            "channel": channel,
                            "work": work,
                            "prim": (row.get("prim") or "").strip(),
                            "cycles": _to_int(row.get("cycles", "0")),
                            "calls": _to_int(row.get("calls", "0")),
                            "pct_tb": _to_float(row.get("pct_tb", "0")),
                            "pct_prim_sum": _to_float(row.get("pct_prim_sum", "0")),
                            "tb_cycles": tb_cycles,
                        }
                    )

    return prim_rows, works


def summarize(prim_rows: List[dict], works: Dict[Tuple[str, int, int], WorkRecord]):
    total_tb_cycles = sum(w.tb_cycles for w in works.values())
    total_prim_cycles = sum(r["cycles"] for r in prim_rows)

    prim_stats = defaultdict(lambda: {"cycles": 0, "calls": 0, "works": 0})
    prim_work_seen = defaultdict(set)

    for r in prim_rows:
        p = r["prim"]
        prim_stats[p]["cycles"] += r["cycles"]
        prim_stats[p]["calls"] += r["calls"]
        prim_work_seen[p].add((r["source"], r["channel"], r["work"]))

    prim_summary = []
    for p, s in prim_stats.items():
        cycles = s["cycles"]
        calls = s["calls"]
        works_hit = len(prim_work_seen[p])
        prim_summary.append(
            {
                "primitive": p,
                "total_cycles": cycles,
                "total_calls": calls,
                "avg_cycles_per_call": (cycles / calls) if calls else 0.0,
                "work_items": works_hit,
                "pct_of_prim_cycles": (100.0 * cycles / total_prim_cycles) if total_prim_cycles else 0.0,
                "pct_of_tb_cycles": (100.0 * cycles / total_tb_cycles) if total_tb_cycles else 0.0,
            }
        )

    prim_summary.sort(key=lambda x: x["total_cycles"], reverse=True)

    channel_stats = defaultdict(lambda: {"tb_cycles": 0, "prim_cycles_total": 0, "works": 0})
    for w in works.values():
        channel_stats[w.channel]["tb_cycles"] += w.tb_cycles
        channel_stats[w.channel]["prim_cycles_total"] += w.prim_cycles_total
        channel_stats[w.channel]["works"] += 1

    channel_summary = []
    for ch, s in sorted(channel_stats.items()):
        tb = s["tb_cycles"]
        prim = s["prim_cycles_total"]
        channel_summary.append(
            {
                "channel": ch,
                "work_items": s["works"],
                "tb_cycles": tb,
                "prim_cycles_total": prim,
                "prim_over_tb_pct": (100.0 * prim / tb) if tb else 0.0,
            }
        )

    file_stats = defaultdict(lambda: {"tb_cycles": 0, "works": 0, "prim_rows": 0, "prim_cycles": 0})
    for w in works.values():
        file_stats[w.source]["tb_cycles"] += w.tb_cycles
        file_stats[w.source]["works"] += 1
    for r in prim_rows:
        file_stats[r["source"]]["prim_rows"] += 1
        file_stats[r["source"]]["prim_cycles"] += r["cycles"]

    file_summary = []
    for src in sorted(file_stats.keys()):
        s = file_stats[src]
        file_summary.append(
            {
                "source": src,
                "work_items": s["works"],
                "tb_cycles": s["tb_cycles"],
                "prim_rows": s["prim_rows"],
                "prim_cycles": s["prim_cycles"],
            }
        )

    top_works = sorted(works.values(), key=lambda w: w.tb_cycles, reverse=True)

    global_summary = {
        "files": len(file_summary),
        "work_items": len(works),
        "prim_rows": len(prim_rows),
        "total_tb_cycles": total_tb_cycles,
        "total_prim_cycles": total_prim_cycles,
        "prim_over_tb_pct": (100.0 * total_prim_cycles / total_tb_cycles) if total_tb_cycles else 0.0,
    }

    return global_summary, prim_summary, channel_summary, file_summary, top_works


def write_csv(path: str, rows: List[dict], fieldnames: List[str]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def try_plot(outdir: str, prim_summary: List[dict], channel_summary: List[dict], topk: int) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as e:
        print(f"[WARN] matplotlib not available, skip plots: {e}")
        return

    top = prim_summary[:topk]
    if top:
        names = [x["primitive"] for x in top][::-1]
        cycles = [x["total_cycles"] for x in top][::-1]
        pct_tb = [x["pct_of_tb_cycles"] for x in top][::-1]

        fig, ax = plt.subplots(figsize=(10, max(4, len(top) * 0.35)))
        ax.barh(names, cycles)
        ax.set_xlabel("Total cycles")
        ax.set_title(f"Top {len(top)} primitives by cycles")
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, "prim_total_cycles_topk.png"), dpi=150)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(10, max(4, len(top) * 0.35)))
        ax.barh(names, pct_tb)
        ax.set_xlabel("Pct of TB cycles (%)")
        ax.set_title(f"Top {len(top)} primitives by TB occupancy")
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, "prim_pct_tb_topk.png"), dpi=150)
        plt.close(fig)

    if channel_summary:
        ch = [str(x["channel"]) for x in channel_summary]
        tb = [x["tb_cycles"] for x in channel_summary]
        prim = [x["prim_cycles_total"] for x in channel_summary]

        fig, ax = plt.subplots(figsize=(10, 4.5))
        ax.bar(ch, tb, label="tb_cycles")
        ax.bar(ch, prim, label="prim_cycles_total")
        ax.set_xlabel("Channel")
        ax.set_ylabel("Cycles")
        ax.set_title("TB cycles vs primitive cycles by channel")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, "channel_tb_vs_prim_cycles.png"), dpi=150)
        plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze NCCL primitive profile CSV files")
    parser.add_argument(
        "--input",
        nargs="+",
        required=True,
        help="Input CSV files or glob patterns, e.g. /tmp/nccl_prim_profile_rank*.csv",
    )
    parser.add_argument("--outdir", default="prim_profile_report", help="Output directory")
    parser.add_argument("--topk", type=int, default=20, help="Top-K primitives to plot")
    parser.add_argument("--no-plots", action="store_true", help="Do not generate PNG plots")
    args = parser.parse_args()

    files = resolve_inputs(args.input)
    if not files:
        print("[ERROR] no input CSV files found", file=sys.stderr)
        return 2

    os.makedirs(args.outdir, exist_ok=True)

    prim_rows, works = load_data(files)
    global_summary, prim_summary, channel_summary, file_summary, top_works = summarize(prim_rows, works)

    write_csv(
        os.path.join(args.outdir, "summary_primitives.csv"),
        prim_summary,
        [
            "primitive",
            "total_cycles",
            "total_calls",
            "avg_cycles_per_call",
            "work_items",
            "pct_of_prim_cycles",
            "pct_of_tb_cycles",
        ],
    )
    write_csv(
        os.path.join(args.outdir, "summary_channels.csv"),
        channel_summary,
        ["channel", "work_items", "tb_cycles", "prim_cycles_total", "prim_over_tb_pct"],
    )
    write_csv(
        os.path.join(args.outdir, "summary_files.csv"),
        file_summary,
        ["source", "work_items", "tb_cycles", "prim_rows", "prim_cycles"],
    )
    write_csv(
        os.path.join(args.outdir, "top_work_items.csv"),
        [
            {
                "source": w.source,
                "channel": w.channel,
                "work": w.work,
                "tb_cycles": w.tb_cycles,
                "prim_cycles_total": w.prim_cycles_total,
                "start_clk": w.start_clk,
                "stop_clk": w.stop_clk,
            }
            for w in top_works[:200]
        ],
        ["source", "channel", "work", "tb_cycles", "prim_cycles_total", "start_clk", "stop_clk"],
    )

    with open(os.path.join(args.outdir, "summary_global.txt"), "w") as f:
        for k, v in global_summary.items():
            f.write(f"{k}: {v}\n")

    if not args.no_plots:
        try_plot(args.outdir, prim_summary, channel_summary, args.topk)

    print("[OK] Input files:", len(files))
    print("[OK] Output dir:", os.path.abspath(args.outdir))
    print("[OK] Work items:", global_summary["work_items"])
    print("[OK] Primitive rows:", global_summary["prim_rows"])
    print("[OK] Total TB cycles:", global_summary["total_tb_cycles"])
    print("[OK] Total primitive cycles:", global_summary["total_prim_cycles"])
    print("[OK] Primitive/TB (%):", f"{global_summary['prim_over_tb_pct']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
