#!/usr/bin/env python3
"""Analyze NCCL primitive profile CSV with TB and channel-lifecycle views.

Input format (from NCCL_PRIM_PROFILE_FILE):
  type,channel,work,tb_cycles,prim_cycles_total,prim,cycles,calls,pct_tb,pct_prim_sum,start_clk,stop_clk,trace_seq,trace_start,trace_stop,trace_dur,trace_start_off,trace_stop_off,trace_dropped

Supported row types:
- tb    : one row per TB/work-item summary
- prim  : one row per TB+primitive aggregate
- trace : one row per primitive invocation with timing
"""

from __future__ import annotations

import argparse
import csv
import glob
import math
import os
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Tuple


@dataclass
class OpStat:
    cycles: int = 0
    calls: int = 0


@dataclass
class TraceEvent:
    primitive: str
    seq: int
    start: int
    stop: int

    @property
    def dur(self) -> int:
        return max(self.stop - self.start, 0)


@dataclass
class TBRecord:
    source: str
    channel: int
    work: int
    tb_cycles: int = 0
    start_clk: int = 0
    stop_clk: int = 0
    trace_dropped: int = 0
    ops: Dict[str, OpStat] = field(default_factory=dict)
    traces: List[TraceEvent] = field(default_factory=list)

    @property
    def busy_cycles(self) -> int:
        if self.traces:
            return sum(t.dur for t in self.traces)
        return sum(v.cycles for v in self.ops.values())

    @property
    def waste_cycles(self) -> int:
        return max(self.tb_cycles - self.busy_cycles, 0)

    @property
    def oversub_cycles(self) -> int:
        return max(self.busy_cycles - self.tb_cycles, 0)

    @property
    def busy_pct(self) -> float:
        return (100.0 * self.busy_cycles / self.tb_cycles) if self.tb_cycles > 0 else 0.0

    @property
    def waste_pct(self) -> float:
        return (100.0 * self.waste_cycles / self.tb_cycles) if self.tb_cycles > 0 else 0.0


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


def percentile(sorted_vals: List[float], p: float) -> float:
    if not sorted_vals:
        return 0.0
    if p <= 0:
        return sorted_vals[0]
    if p >= 100:
        return sorted_vals[-1]
    idx = (len(sorted_vals) - 1) * (p / 100.0)
    lo = int(math.floor(idx))
    hi = int(math.ceil(idx))
    if lo == hi:
        return sorted_vals[lo]
    w = idx - lo
    return sorted_vals[lo] * (1.0 - w) + sorted_vals[hi] * w


def resolve_inputs(patterns: Iterable[str]) -> List[str]:
    paths: List[str] = []
    for p in patterns:
        expanded = os.path.expanduser(p)
        matched = sorted(glob.glob(expanded))
        if matched:
            paths.extend(matched)
        elif os.path.isfile(expanded):
            paths.append(expanded)

    seen = set()
    uniq: List[str] = []
    for p in paths:
        if p in seen:
            continue
        seen.add(p)
        uniq.append(p)
    return uniq


def _interval_union_length(intervals: List[Tuple[int, int]]) -> int:
    if not intervals:
        return 0
    ordered = sorted((s, e) for s, e in intervals if e > s)
    if not ordered:
        return 0
    total = 0
    cur_s, cur_e = ordered[0]
    for s, e in ordered[1:]:
        if s > cur_e:
            total += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    total += cur_e - cur_s
    return total


def load_tb_records(files: List[str]) -> Dict[Tuple[str, int, int], TBRecord]:
    records: Dict[Tuple[str, int, int], TBRecord] = {}

    for src in files:
        with open(src, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                typ = (row.get("type") or "").strip()
                if typ not in ("tb", "prim", "trace"):
                    continue

                channel = _to_int(row.get("channel", "0"))
                work = _to_int(row.get("work", "0"))
                key = (src, channel, work)

                rec = records.get(key)
                if rec is None:
                    rec = TBRecord(source=src, channel=channel, work=work)
                    records[key] = rec

                tb_cycles = _to_int(row.get("tb_cycles", "0"))
                start_clk = _to_int(row.get("start_clk", "0"))
                stop_clk = _to_int(row.get("stop_clk", "0"))
                rec.trace_dropped = max(rec.trace_dropped, _to_int(row.get("trace_dropped", "0")))

                if tb_cycles > rec.tb_cycles:
                    rec.tb_cycles = tb_cycles
                if rec.start_clk == 0 and start_clk > 0:
                    rec.start_clk = start_clk
                if stop_clk > rec.stop_clk:
                    rec.stop_clk = stop_clk

                if typ == "prim":
                    prim = (row.get("prim") or "").strip()
                    if not prim:
                        continue
                    st = rec.ops.get(prim)
                    if st is None:
                        st = OpStat()
                        rec.ops[prim] = st
                    st.cycles += _to_int(row.get("cycles", "0"))
                    st.calls += _to_int(row.get("calls", "0"))

                if typ == "trace":
                    prim = (row.get("prim") or "").strip()
                    seq = _to_int(row.get("trace_seq", "0"))
                    t_start = _to_int(row.get("trace_start", "0"))
                    t_stop = _to_int(row.get("trace_stop", "0"))

                    if t_start == 0:
                        t_start = start_clk
                    if t_stop == 0:
                        t_stop = stop_clk

                    if prim and t_stop > t_start:
                        rec.traces.append(TraceEvent(primitive=prim, seq=seq, start=t_start, stop=t_stop))

    for rec in records.values():
        rec.traces.sort(key=lambda x: (x.start, x.seq))
        if rec.traces and rec.stop_clk <= rec.start_clk:
            rec.start_clk = rec.traces[0].start
            rec.stop_clk = rec.traces[-1].stop
            if rec.tb_cycles <= 0 and rec.stop_clk > rec.start_clk:
                rec.tb_cycles = rec.stop_clk - rec.start_clk

    return records


def build_outputs(records: Dict[Tuple[str, int, int], TBRecord]):
    tbs = list(records.values())

    tb_rows: List[dict] = []
    tb_op_rows: List[dict] = []
    tb_trace_rows: List[dict] = []
    channel_rows: List[dict] = []
    channel_op_rows: List[dict] = []

    op_agg = defaultdict(lambda: {"cycles": 0, "calls": 0, "tb_hits": 0})
    file_agg = defaultdict(lambda: {"tb_count": 0, "tb_cycles": 0, "busy_cycles": 0, "waste_cycles": 0})
    channel_agg: Dict[Tuple[str, int], dict] = {}

    total_tb_cycles = 0
    total_busy_cycles = 0
    total_waste_cycles = 0
    total_oversub_cycles = 0

    for rec in tbs:
        op_names = sorted(rec.ops.keys())
        if not op_names and rec.traces:
            op_names = sorted({t.primitive for t in rec.traces})

        total_calls = sum(v.calls for v in rec.ops.values())
        tb_rows.append(
            {
                "source": rec.source,
                "source_base": os.path.basename(rec.source),
                "channel": rec.channel,
                "work": rec.work,
                "tb_cycles": rec.tb_cycles,
                "busy_cycles": rec.busy_cycles,
                "waste_cycles": rec.waste_cycles,
                "oversub_cycles": rec.oversub_cycles,
                "busy_pct": rec.busy_pct,
                "waste_pct": rec.waste_pct,
                "operator_count": len(op_names),
                "operators": "|".join(op_names),
                "total_calls": total_calls,
                "trace_count": len(rec.traces),
                "trace_dropped": rec.trace_dropped,
                "start_clk": rec.start_clk,
                "stop_clk": rec.stop_clk,
            }
        )

        total_tb_cycles += rec.tb_cycles
        total_busy_cycles += rec.busy_cycles
        total_waste_cycles += rec.waste_cycles
        total_oversub_cycles += rec.oversub_cycles

        fa = file_agg[rec.source]
        fa["tb_count"] += 1
        fa["tb_cycles"] += rec.tb_cycles
        fa["busy_cycles"] += rec.busy_cycles
        fa["waste_cycles"] += rec.waste_cycles

        ch_key = (rec.source, rec.channel)
        ch = channel_agg.get(ch_key)
        if ch is None:
            ch = {
                "source": rec.source,
                "source_base": os.path.basename(rec.source),
                "channel": rec.channel,
                "works": set(),
                "tb_count": 0,
                "tb_cycles_sum": 0,
                "busy_cycles_sum": 0,
                "start_clk": 0,
                "stop_clk": 0,
                "trace_count": 0,
                "trace_dropped": 0,
                "trace_intervals": [],
                "op_cycles": defaultdict(int),
                "op_calls": defaultdict(int),
            }
            channel_agg[ch_key] = ch

        ch["works"].add(rec.work)
        ch["tb_count"] += 1
        ch["tb_cycles_sum"] += rec.tb_cycles
        ch["busy_cycles_sum"] += rec.busy_cycles
        ch["trace_count"] += len(rec.traces)
        ch["trace_dropped"] += rec.trace_dropped
        if rec.start_clk > 0 and (ch["start_clk"] == 0 or rec.start_clk < ch["start_clk"]):
            ch["start_clk"] = rec.start_clk
        if rec.stop_clk > ch["stop_clk"]:
            ch["stop_clk"] = rec.stop_clk

        for prim, st in rec.ops.items():
            tb_op_rows.append(
                {
                    "source": rec.source,
                    "source_base": os.path.basename(rec.source),
                    "channel": rec.channel,
                    "work": rec.work,
                    "primitive": prim,
                    "cycles": st.cycles,
                    "calls": st.calls,
                    "tb_cycles": rec.tb_cycles,
                    "pct_tb": (100.0 * st.cycles / rec.tb_cycles) if rec.tb_cycles else 0.0,
                }
            )
            ch["op_cycles"][prim] += st.cycles
            ch["op_calls"][prim] += st.calls
            op_agg[prim]["cycles"] += st.cycles
            op_agg[prim]["calls"] += st.calls
            op_agg[prim]["tb_hits"] += 1

        if rec.traces and not rec.ops:
            tmp_cycles = defaultdict(int)
            tmp_calls = defaultdict(int)
            for t in rec.traces:
                tmp_cycles[t.primitive] += t.dur
                tmp_calls[t.primitive] += 1
            for prim, cyc in tmp_cycles.items():
                ch["op_cycles"][prim] += cyc
                ch["op_calls"][prim] += tmp_calls[prim]
                op_agg[prim]["cycles"] += cyc
                op_agg[prim]["calls"] += tmp_calls[prim]
                op_agg[prim]["tb_hits"] += 1

        for t in rec.traces:
            ch["trace_intervals"].append((t.start, t.stop))
            tb_trace_rows.append(
                {
                    "source": rec.source,
                    "source_base": os.path.basename(rec.source),
                    "channel": rec.channel,
                    "work": rec.work,
                    "tb_start_clk": rec.start_clk,
                    "tb_stop_clk": rec.stop_clk,
                    "tb_cycles": rec.tb_cycles,
                    "primitive": t.primitive,
                    "seq": t.seq,
                    "trace_start": t.start,
                    "trace_stop": t.stop,
                    "trace_dur": t.dur,
                    "start_off": t.start - rec.start_clk,
                    "stop_off": t.stop - rec.start_clk,
                    "pct_tb": (100.0 * t.dur / rec.tb_cycles) if rec.tb_cycles else 0.0,
                }
            )

    tb_rows.sort(key=lambda x: (x["waste_cycles"], x["tb_cycles"]), reverse=True)

    total_channel_cycles = 0
    total_channel_busy_cycles = 0
    total_channel_waste_cycles = 0
    total_channel_oversub_cycles = 0
    channel_busy_pcts: List[float] = []
    channel_waste_pcts: List[float] = []

    for _, ch in sorted(channel_agg.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        channel_cycles = (ch["stop_clk"] - ch["start_clk"]) if ch["stop_clk"] > ch["start_clk"] else ch["tb_cycles_sum"]
        busy_union = _interval_union_length(ch["trace_intervals"]) if ch["trace_intervals"] else ch["busy_cycles_sum"]
        busy_sum = ch["busy_cycles_sum"]
        waste_cycles = max(channel_cycles - busy_union, 0)
        oversub_cycles = max(busy_sum - busy_union, 0)
        busy_pct = (100.0 * busy_union / channel_cycles) if channel_cycles else 0.0
        waste_pct = (100.0 * waste_cycles / channel_cycles) if channel_cycles else 0.0

        operators = sorted(ch["op_cycles"].keys())
        channel_rows.append(
            {
                "source": ch["source"],
                "source_base": ch["source_base"],
                "channel": ch["channel"],
                "work_count": len(ch["works"]),
                "tb_count": ch["tb_count"],
                "channel_start_clk": ch["start_clk"],
                "channel_stop_clk": ch["stop_clk"],
                "channel_cycles": channel_cycles,
                "tb_cycles_sum": ch["tb_cycles_sum"],
                "busy_cycles": busy_union,
                "busy_cycles_sum": busy_sum,
                "waste_cycles": waste_cycles,
                "oversub_cycles": oversub_cycles,
                "busy_pct": busy_pct,
                "waste_pct": waste_pct,
                "trace_count": ch["trace_count"],
                "trace_dropped": ch["trace_dropped"],
                "operator_count": len(operators),
                "operators": "|".join(operators),
            }
        )

        total_channel_cycles += channel_cycles
        total_channel_busy_cycles += busy_union
        total_channel_waste_cycles += waste_cycles
        total_channel_oversub_cycles += oversub_cycles
        if channel_cycles > 0:
            channel_busy_pcts.append(busy_pct)
            channel_waste_pcts.append(waste_pct)

        for prim in operators:
            cyc = ch["op_cycles"][prim]
            calls = ch["op_calls"][prim]
            channel_op_rows.append(
                {
                    "source": ch["source"],
                    "source_base": ch["source_base"],
                    "channel": ch["channel"],
                    "primitive": prim,
                    "cycles": cyc,
                    "calls": calls,
                    "pct_channel_busy": (100.0 * cyc / busy_union) if busy_union else 0.0,
                    "pct_channel_lifecycle": (100.0 * cyc / channel_cycles) if channel_cycles else 0.0,
                }
            )

    channel_rows.sort(key=lambda x: (x["waste_cycles"], x["channel_cycles"]), reverse=True)
    channel_op_rows.sort(key=lambda x: (x["source_base"], x["channel"], -x["cycles"]))

    op_rows = []
    for prim, s in op_agg.items():
        cycles = s["cycles"]
        calls = s["calls"]
        op_rows.append(
            {
                "primitive": prim,
                "total_cycles": cycles,
                "total_calls": calls,
                "tb_hits": s["tb_hits"],
                "avg_cycles_per_call": (cycles / calls) if calls else 0.0,
                "avg_cycles_per_tb_hit": (cycles / s["tb_hits"]) if s["tb_hits"] else 0.0,
                "pct_of_busy_cycles": (100.0 * cycles / total_busy_cycles) if total_busy_cycles else 0.0,
                "pct_of_tb_cycles": (100.0 * cycles / total_tb_cycles) if total_tb_cycles else 0.0,
                "pct_of_channel_busy_cycles": (100.0 * cycles / total_channel_busy_cycles) if total_channel_busy_cycles else 0.0,
                "pct_of_channel_cycles": (100.0 * cycles / total_channel_cycles) if total_channel_cycles else 0.0,
            }
        )
    op_rows.sort(key=lambda x: x["total_cycles"], reverse=True)

    file_rows = []
    for src in sorted(file_agg.keys()):
        s = file_agg[src]
        file_rows.append(
            {
                "source": src,
                "source_base": os.path.basename(src),
                "tb_count": s["tb_count"],
                "tb_cycles": s["tb_cycles"],
                "busy_cycles": s["busy_cycles"],
                "waste_cycles": s["waste_cycles"],
                "busy_pct": (100.0 * s["busy_cycles"] / s["tb_cycles"]) if s["tb_cycles"] else 0.0,
                "waste_pct": (100.0 * s["waste_cycles"] / s["tb_cycles"]) if s["tb_cycles"] else 0.0,
            }
        )

    busy_pct_vals = sorted([x["busy_pct"] for x in tb_rows if x["tb_cycles"] > 0])
    waste_pct_vals = sorted([x["waste_pct"] for x in tb_rows if x["tb_cycles"] > 0])
    channel_busy_pcts.sort()
    channel_waste_pcts.sort()

    global_summary = {
        "files": len(file_rows),
        "tb_count": len(tb_rows),
        "channel_count": len(channel_rows),
        "tb_with_ops": sum(1 for x in tb_rows if x["operator_count"] > 0),
        "tb_without_ops": sum(1 for x in tb_rows if x["operator_count"] == 0),
        "tb_with_trace": sum(1 for x in tb_rows if x["trace_count"] > 0),
        "trace_rows": len(tb_trace_rows),
        "total_tb_cycles": total_tb_cycles,
        "total_busy_cycles": total_busy_cycles,
        "total_waste_cycles": total_waste_cycles,
        "total_oversub_cycles": total_oversub_cycles,
        "global_busy_pct": (100.0 * total_busy_cycles / total_tb_cycles) if total_tb_cycles else 0.0,
        "global_waste_pct": (100.0 * total_waste_cycles / total_tb_cycles) if total_tb_cycles else 0.0,
        "busy_pct_p50": percentile(busy_pct_vals, 50),
        "busy_pct_p90": percentile(busy_pct_vals, 90),
        "busy_pct_p99": percentile(busy_pct_vals, 99),
        "waste_pct_p50": percentile(waste_pct_vals, 50),
        "waste_pct_p90": percentile(waste_pct_vals, 90),
        "waste_pct_p99": percentile(waste_pct_vals, 99),
        "total_channel_cycles": total_channel_cycles,
        "total_channel_busy_cycles": total_channel_busy_cycles,
        "total_channel_waste_cycles": total_channel_waste_cycles,
        "total_channel_oversub_cycles": total_channel_oversub_cycles,
        "global_channel_busy_pct": (100.0 * total_channel_busy_cycles / total_channel_cycles) if total_channel_cycles else 0.0,
        "global_channel_waste_pct": (100.0 * total_channel_waste_cycles / total_channel_cycles) if total_channel_cycles else 0.0,
        "channel_busy_pct_p50": percentile(channel_busy_pcts, 50),
        "channel_busy_pct_p90": percentile(channel_busy_pcts, 90),
        "channel_busy_pct_p99": percentile(channel_busy_pcts, 99),
        "channel_waste_pct_p50": percentile(channel_waste_pcts, 50),
        "channel_waste_pct_p90": percentile(channel_waste_pcts, 90),
        "channel_waste_pct_p99": percentile(channel_waste_pcts, 99),
    }

    return global_summary, tb_rows, tb_op_rows, tb_trace_rows, channel_rows, channel_op_rows, op_rows, file_rows


def write_csv(path: str, rows: List[dict], fieldnames: List[str]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def _channel_timeline_plot_by_source(outdir: str, records_by_source: Dict[str, List[TBRecord]], topk: int) -> None:
    import matplotlib.pyplot as plt  # type: ignore
    import matplotlib.patches as mpatches  # type: ignore

    cmap = plt.get_cmap("tab20")

    for source, tbs in records_by_source.items():
        ch_events: Dict[int, List[Tuple[int, int, str]]] = defaultdict(list)
        for tb in tbs:
            for tr in tb.traces:
                ch_events[tb.channel].append((tr.start, tr.stop, tr.primitive))

        ch_events = {ch: evs for ch, evs in ch_events.items() if evs}
        if not ch_events:
            continue

        channels = sorted(ch_events.keys())
        if topk > 0 and len(channels) > topk:
            channels = sorted(
                channels,
                key=lambda ch: sum(max(e - s, 0) for s, e, _ in ch_events[ch]),
                reverse=True,
            )[:topk]
            channels.sort()

        min_start = min(s for ch in channels for s, _, _ in ch_events[ch])
        max_stop = max(e for ch in channels for _, e, _ in ch_events[ch])
        prims = sorted({p for ch in channels for _, _, p in ch_events[ch]})
        color_of = {p: cmap(i % 20) for i, p in enumerate(prims)}

        fig_h = max(4, 0.45 * len(channels) + 1.5)
        fig, ax = plt.subplots(figsize=(14, fig_h))

        for i, ch in enumerate(channels):
            events = sorted(ch_events[ch], key=lambda x: (x[0], x[1]))
            for s, e, prim in events:
                dur = e - s
                if dur <= 0:
                    continue
                ax.broken_barh([(s - min_start, dur)], (i - 0.35, 0.7), facecolors=color_of[prim], edgecolors="none")

        ax.set_yticks(list(range(len(channels))))
        ax.set_yticklabels([f"ch{ch}" for ch in channels])
        ax.set_xlabel("GPU clock offset (ticks)")
        ax.set_ylabel("Channel (all works merged)")
        ax.set_title(f"Channel primitive timeline ({os.path.basename(source)})")
        ax.set_xlim(0, max_stop - min_start)
        ax.grid(axis="x", linestyle=":", alpha=0.4)

        handles = [mpatches.Patch(color=color_of[p], label=p) for p in prims[:20]]
        if handles:
            ax.legend(handles=handles, loc="upper right", fontsize=8, ncol=1)

        fig.tight_layout()
        out = os.path.join(outdir, f"channel_timeline_{os.path.basename(source)}.png")
        fig.savefig(out, dpi=160)
        plt.close(fig)


def try_plot(
    outdir: str,
    records: Dict[Tuple[str, int, int], TBRecord],
    tb_rows: List[dict],
    channel_rows: List[dict],
    op_rows: List[dict],
    topk: int,
) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as e:
        print(f"[WARN] matplotlib not available, skip plots: {e}")
        return

    busy_vals = [x["busy_pct"] for x in tb_rows if x["tb_cycles"] > 0]
    if busy_vals:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.hist(busy_vals, bins=30)
        ax.set_xlabel("TB busy ratio (%)")
        ax.set_ylabel("TB count")
        ax.set_title("TB busy ratio distribution")
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, "tb_busy_ratio_hist.png"), dpi=150)
        plt.close(fig)

    if channel_rows:
        ch_sorted = sorted(channel_rows, key=lambda x: (x["source_base"], x["channel"]))
        labels = [f"{x['source_base']}:ch{x['channel']}" for x in ch_sorted]
        busy = [x["busy_cycles"] for x in ch_sorted]
        waste = [x["waste_cycles"] for x in ch_sorted]

        fig, ax = plt.subplots(figsize=(max(10, 0.5 * len(labels)), 4.8))
        ax.bar(labels, busy, label="busy_cycles")
        ax.bar(labels, waste, bottom=busy, label="waste_cycles")
        ax.set_xlabel("Source:Channel")
        ax.set_ylabel("Cycles")
        ax.set_title("Channel lifecycle: busy vs waste (works merged)")
        ax.legend()
        ax.tick_params(axis="x", rotation=60)
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, "channel_busy_waste_stacked.png"), dpi=150)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(max(10, 0.5 * len(labels)), 4.8))
        ax.bar(labels, [x["waste_pct"] for x in ch_sorted])
        ax.set_xlabel("Source:Channel")
        ax.set_ylabel("Waste pct (%)")
        ax.set_title("Channel waste ratio (works merged)")
        ax.tick_params(axis="x", rotation=60)
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, "channel_waste_pct.png"), dpi=150)
        plt.close(fig)

        top_ch = sorted(channel_rows, key=lambda x: x["waste_cycles"], reverse=True)[:topk]
        if top_ch:
            y_labels = [f"{x['source_base']}:ch{x['channel']}" for x in top_ch][::-1]
            y_busy = [x["busy_cycles"] for x in top_ch][::-1]
            y_waste = [x["waste_cycles"] for x in top_ch][::-1]
            fig, ax = plt.subplots(figsize=(12, max(4, 0.35 * len(top_ch))))
            ax.barh(y_labels, y_busy, label="busy_cycles")
            ax.barh(y_labels, y_waste, left=y_busy, label="waste_cycles")
            ax.set_xlabel("Cycles")
            ax.set_title(f"Top {len(top_ch)} channels by waste")
            ax.legend()
            fig.tight_layout()
            fig.savefig(os.path.join(outdir, "channel_topk_waste_stacked.png"), dpi=150)
            plt.close(fig)

    top_op = op_rows[:topk]
    if top_op:
        names = [x["primitive"] for x in top_op][::-1]
        vals = [x["pct_of_channel_busy_cycles"] for x in top_op][::-1]
        fig, ax = plt.subplots(figsize=(10, max(4, 0.35 * len(top_op))))
        ax.barh(names, vals)
        ax.set_xlabel("Pct of channel busy cycles (%)")
        ax.set_title(f"Top {len(top_op)} operators by busy-cycle contribution")
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, "operator_busy_contrib_topk.png"), dpi=150)
        plt.close(fig)

    records_by_source: Dict[str, List[TBRecord]] = defaultdict(list)
    for rec in records.values():
        records_by_source[rec.source].append(rec)
    _channel_timeline_plot_by_source(outdir, records_by_source, topk)


def main() -> int:
    parser = argparse.ArgumentParser(description="NCCL primitive profile analyzer")
    parser.add_argument(
        "--input",
        nargs="+",
        required=True,
        help="Input CSV files or glob patterns, e.g. /tmp/nccl_prim_profile_rank*.csv",
    )
    parser.add_argument("--outdir", default="prim_profile_report", help="Output directory")
    parser.add_argument("--topk", type=int, default=30, help="Top-K rows for ranking plots/channels in timeline")
    parser.add_argument("--no-plots", action="store_true", help="Disable PNG plot generation")
    args = parser.parse_args()

    files = resolve_inputs(args.input)
    if not files:
        print("[ERROR] no input CSV files found", file=sys.stderr)
        return 2

    os.makedirs(args.outdir, exist_ok=True)

    records = load_tb_records(files)
    global_summary, tb_rows, tb_op_rows, tb_trace_rows, channel_rows, channel_op_rows, op_rows, file_rows = build_outputs(records)

    write_csv(
        os.path.join(args.outdir, "tb_breakdown.csv"),
        tb_rows,
        [
            "source",
            "source_base",
            "channel",
            "work",
            "tb_cycles",
            "busy_cycles",
            "waste_cycles",
            "oversub_cycles",
            "busy_pct",
            "waste_pct",
            "operator_count",
            "operators",
            "total_calls",
            "trace_count",
            "trace_dropped",
            "start_clk",
            "stop_clk",
        ],
    )

    write_csv(
        os.path.join(args.outdir, "tb_operator_rows.csv"),
        tb_op_rows,
        [
            "source",
            "source_base",
            "channel",
            "work",
            "primitive",
            "cycles",
            "calls",
            "tb_cycles",
            "pct_tb",
        ],
    )

    write_csv(
        os.path.join(args.outdir, "tb_trace_rows.csv"),
        tb_trace_rows,
        [
            "source",
            "source_base",
            "channel",
            "work",
            "tb_start_clk",
            "tb_stop_clk",
            "tb_cycles",
            "primitive",
            "seq",
            "trace_start",
            "trace_stop",
            "trace_dur",
            "start_off",
            "stop_off",
            "pct_tb",
        ],
    )

    channel_fields = [
        "source",
        "source_base",
        "channel",
        "work_count",
        "tb_count",
        "channel_start_clk",
        "channel_stop_clk",
        "channel_cycles",
        "tb_cycles_sum",
        "busy_cycles",
        "busy_cycles_sum",
        "waste_cycles",
        "oversub_cycles",
        "busy_pct",
        "waste_pct",
        "trace_count",
        "trace_dropped",
        "operator_count",
        "operators",
    ]
    write_csv(os.path.join(args.outdir, "channel_breakdown.csv"), channel_rows, channel_fields)
    write_csv(os.path.join(args.outdir, "summary_channels.csv"), channel_rows, channel_fields)

    write_csv(
        os.path.join(args.outdir, "channel_operator_rows.csv"),
        channel_op_rows,
        [
            "source",
            "source_base",
            "channel",
            "primitive",
            "cycles",
            "calls",
            "pct_channel_busy",
            "pct_channel_lifecycle",
        ],
    )

    write_csv(
        os.path.join(args.outdir, "summary_operators.csv"),
        op_rows,
        [
            "primitive",
            "total_cycles",
            "total_calls",
            "tb_hits",
            "avg_cycles_per_call",
            "avg_cycles_per_tb_hit",
            "pct_of_busy_cycles",
            "pct_of_tb_cycles",
            "pct_of_channel_busy_cycles",
            "pct_of_channel_cycles",
        ],
    )

    write_csv(
        os.path.join(args.outdir, "summary_files.csv"),
        file_rows,
        [
            "source",
            "source_base",
            "tb_count",
            "tb_cycles",
            "busy_cycles",
            "waste_cycles",
            "busy_pct",
            "waste_pct",
        ],
    )

    with open(os.path.join(args.outdir, "summary_global.txt"), "w") as f:
        for k, v in global_summary.items():
            f.write(f"{k}: {v}\n")

    if not args.no_plots:
        try_plot(args.outdir, records, tb_rows, channel_rows, op_rows, args.topk)

    print("[OK] Input files:", len(files))
    print("[OK] Output dir:", os.path.abspath(args.outdir))
    print("[OK] TB count:", global_summary["tb_count"])
    print("[OK] Channel count:", global_summary["channel_count"])
    print("[OK] TB with trace:", global_summary["tb_with_trace"])
    print("[OK] Trace rows:", global_summary["trace_rows"])
    print("[OK] Total channel cycles:", global_summary["total_channel_cycles"])
    print("[OK] Total channel busy cycles:", global_summary["total_channel_busy_cycles"])
    print("[OK] Total channel waste cycles:", global_summary["total_channel_waste_cycles"])
    print(
        "[OK] Global channel busy/waste (%):",
        f"{global_summary['global_channel_busy_pct']:.4f}/{global_summary['global_channel_waste_pct']:.4f}",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
