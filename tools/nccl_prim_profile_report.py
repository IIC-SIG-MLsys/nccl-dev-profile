#!/usr/bin/env python3
"""Focused NCCL primitive profile analyzer.

Goal:
- Inspect one NCCL call (`op_count`) at a time.
- Show TB lifecycle, primitive distribution, and idle/non-primitive gaps.
- Keep outputs minimal and directly useful for diagnosing where time is spent.

Supported CSV columns:
  type,op_count,channel,work,tb_cycles,prim_cycles_total,wait_cycles,
  compute_cycles,sync_cycles,prim,cycles,calls,pct_tb,pct_prim_sum,start_clk,
  stop_clk,trace_group,trace_seq,trace_start,trace_stop,trace_dur,
  trace_start_off,trace_stop_off,trace_dropped

Legacy CSVs without `op_count` / `trace_group` are also supported.
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple


@dataclass
class PrimStat:
    cycles: int = 0
    calls: int = 0


@dataclass
class TraceEvent:
    primitive: str
    group: int
    seq: int
    start: int
    stop: int

    @property
    def dur(self) -> int:
        return max(self.stop - self.start, 0)


@dataclass
class TBRecord:
    source: str
    op_count: int
    channel: int
    work: int
    tb_cycles: int = 0
    prim_cycles_total: int = 0
    wait_cycles: int = 0
    compute_cycles: int = 0
    sync_cycles: int = 0
    start_clk: int = 0
    stop_clk: int = 0
    trace_dropped: int = 0
    prims: Dict[str, PrimStat] = field(default_factory=dict)
    traces: List[TraceEvent] = field(default_factory=list)

    @property
    def busy_cycles(self) -> int:
        if self.prim_cycles_total > 0:
            return self.prim_cycles_total
        return sum(p.cycles for p in self.prims.values())

    @property
    def compute_cycles_derived(self) -> int:
        if self.busy_cycles <= 0:
            return 0
        classified = self.wait_cycles + self.sync_cycles
        if self.compute_cycles > 0:
            return max(self.compute_cycles, self.busy_cycles - classified)
        return max(self.busy_cycles - classified, 0)

    @property
    def idle_cycles(self) -> int:
        return max(self.tb_cycles - self.busy_cycles, 0)

    @property
    def idle_pct(self) -> float:
        return (100.0 * self.idle_cycles / self.tb_cycles) if self.tb_cycles else 0.0

    @property
    def busy_pct(self) -> float:
        return (100.0 * self.busy_cycles / self.tb_cycles) if self.tb_cycles else 0.0

    @property
    def trace_complete(self) -> bool:
        return self.trace_dropped == 0


@dataclass
class GapEvent:
    source: str
    op_count: int
    channel: int
    work: int
    gap_kind: str
    start: int
    stop: int
    dur: int
    prev_primitive: str
    next_primitive: str
    reason_hint: str
    trace_complete: int


REASON_BEFORE_FIRST = "before first primitive: likely primitive setup, connector sync, pointer exchange, or barrier before entering the first primitive"
REASON_BETWEEN = "between primitives: likely primitive boundary overhead, barrier/postPeer bookkeeping, or setup for the next primitive; peer/data waits are usually counted inside the primitive itself"
REASON_AFTER_LAST = "after last primitive: likely primitive teardown, final sync, destructor wait, or kernel epilogue before TB exit"
REASON_INCOMPLETE = "trace incomplete: gap may be under-estimated because some primitive calls were dropped"


def _to_int(value: object) -> int:
    if value is None:
        return 0
    s = str(value).strip()
    if not s:
        return 0
    try:
        return int(float(s))
    except ValueError:
        return 0


def _parse_int_auto(value: str) -> int:
    return int(value, 0)


def resolve_inputs(patterns: Iterable[str]) -> List[str]:
    out: List[str] = []
    for pattern in patterns:
        expanded = os.path.expanduser(pattern)
        matches = sorted(glob.glob(expanded))
        if matches:
            out.extend(matches)
        elif os.path.isfile(expanded):
            out.append(expanded)

    seen = set()
    uniq: List[str] = []
    for path in out:
        if path in seen:
            continue
        seen.add(path)
        uniq.append(path)
    return uniq


def load_records(files: List[str], op_filter: Optional[set[int]]) -> Dict[Tuple[str, int, int, int], TBRecord]:
    records: Dict[Tuple[str, int, int, int], TBRecord] = {}
    for src in files:
        with open(src, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                typ = (row.get("type") or "").strip()
                if typ not in {"tb", "prim", "trace"}:
                    continue

                op_count = _to_int(row.get("op_count", "0"))
                if op_filter is not None and op_count not in op_filter:
                    continue

                channel = _to_int(row.get("channel", "0"))
                work = _to_int(row.get("work", "0"))
                key = (src, op_count, channel, work)
                rec = records.get(key)
                if rec is None:
                    rec = TBRecord(source=src, op_count=op_count, channel=channel, work=work)
                    records[key] = rec

                rec.tb_cycles = max(rec.tb_cycles, _to_int(row.get("tb_cycles", "0")))
                rec.prim_cycles_total = max(rec.prim_cycles_total, _to_int(row.get("prim_cycles_total", "0")))
                rec.wait_cycles = max(rec.wait_cycles, _to_int(row.get("wait_cycles", "0")))
                rec.compute_cycles = max(rec.compute_cycles, _to_int(row.get("compute_cycles", "0")))
                rec.sync_cycles = max(rec.sync_cycles, _to_int(row.get("sync_cycles", "0")))
                rec.trace_dropped = max(rec.trace_dropped, _to_int(row.get("trace_dropped", "0")))

                start_clk = _to_int(row.get("start_clk", "0"))
                stop_clk = _to_int(row.get("stop_clk", "0"))
                if rec.start_clk == 0 and start_clk > 0:
                    rec.start_clk = start_clk
                if stop_clk > rec.stop_clk:
                    rec.stop_clk = stop_clk

                if typ == "prim":
                    primitive = (row.get("prim") or "").strip()
                    if primitive:
                        st = rec.prims.get(primitive)
                        if st is None:
                            st = PrimStat()
                            rec.prims[primitive] = st
                        st.cycles += _to_int(row.get("cycles", "0"))
                        st.calls += _to_int(row.get("calls", "0"))
                elif typ == "trace":
                    primitive = (row.get("prim") or "").strip()
                    t_start = _to_int(row.get("trace_start", "0")) or start_clk
                    t_stop = _to_int(row.get("trace_stop", "0")) or stop_clk
                    if primitive and t_stop > t_start:
                        rec.traces.append(
                            TraceEvent(
                                primitive=primitive,
                                group=_to_int(row.get("trace_group", "0")),
                                seq=_to_int(row.get("trace_seq", "0")),
                                start=t_start,
                                stop=t_stop,
                            )
                        )

    for rec in records.values():
        rec.traces.sort(key=lambda x: (x.start, x.group, x.seq))
        if rec.traces and rec.stop_clk <= rec.start_clk:
            rec.start_clk = rec.traces[0].start
            rec.stop_clk = max(t.stop for t in rec.traces)
            if rec.tb_cycles <= 0:
                rec.tb_cycles = max(rec.stop_clk - rec.start_clk, 0)
    return records


def pick_default_calls(records: Iterable[TBRecord], requested: Optional[set[int]], top_calls: int) -> List[int]:
    if requested:
        return sorted(requested)
    counts: Dict[int, int] = defaultdict(int)
    for rec in records:
        counts[rec.op_count] += 1
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return [op for op, _ in ranked[:top_calls]]


def compute_gaps(rec: TBRecord) -> List[GapEvent]:
    gaps: List[GapEvent] = []
    traces = sorted(rec.traces, key=lambda x: (x.start, x.group, x.seq))
    if rec.stop_clk <= rec.start_clk:
        return gaps

    if not traces:
        if rec.tb_cycles > 0:
            gaps.append(
                GapEvent(
                    source=rec.source,
                    op_count=rec.op_count,
                    channel=rec.channel,
                    work=rec.work,
                    gap_kind="full_tb_without_trace",
                    start=rec.start_clk,
                    stop=rec.stop_clk,
                    dur=rec.tb_cycles,
                    prev_primitive="",
                    next_primitive="",
                    reason_hint=REASON_INCOMPLETE if rec.trace_dropped > 0 else "no primitive trace present; rely on aggregated primitive cycles instead",
                    trace_complete=1 if rec.trace_complete else 0,
                )
            )
        return gaps

    first = traces[0]
    if first.start > rec.start_clk:
        gaps.append(
            GapEvent(
                source=rec.source,
                op_count=rec.op_count,
                channel=rec.channel,
                work=rec.work,
                gap_kind="before_first_primitive",
                start=rec.start_clk,
                stop=first.start,
                dur=first.start - rec.start_clk,
                prev_primitive="",
                next_primitive=first.primitive,
                reason_hint=REASON_BEFORE_FIRST if rec.trace_complete else f"{REASON_BEFORE_FIRST}; {REASON_INCOMPLETE}",
                trace_complete=1 if rec.trace_complete else 0,
            )
        )

    prev = first
    for cur in traces[1:]:
        if cur.start > prev.stop:
            gaps.append(
                GapEvent(
                    source=rec.source,
                    op_count=rec.op_count,
                    channel=rec.channel,
                    work=rec.work,
                    gap_kind="between_primitives",
                    start=prev.stop,
                    stop=cur.start,
                    dur=cur.start - prev.stop,
                    prev_primitive=prev.primitive,
                    next_primitive=cur.primitive,
                    reason_hint=REASON_BETWEEN if rec.trace_complete else f"{REASON_BETWEEN}; {REASON_INCOMPLETE}",
                    trace_complete=1 if rec.trace_complete else 0,
                )
            )
        if cur.stop > prev.stop:
            prev = cur

    last = max(traces, key=lambda x: x.stop)
    if rec.stop_clk > last.stop:
        gaps.append(
            GapEvent(
                source=rec.source,
                op_count=rec.op_count,
                channel=rec.channel,
                work=rec.work,
                gap_kind="after_last_primitive",
                start=last.stop,
                stop=rec.stop_clk,
                dur=rec.stop_clk - last.stop,
                prev_primitive=last.primitive,
                next_primitive="",
                reason_hint=REASON_AFTER_LAST if rec.trace_complete else f"{REASON_AFTER_LAST}; {REASON_INCOMPLETE}",
                trace_complete=1 if rec.trace_complete else 0,
            )
        )

    return [g for g in gaps if g.dur > 0]


def classify_tb_wait(rec: TBRecord, gaps: List[GapEvent]) -> str:
    largest_gap = max(gaps, key=lambda x: x.dur, default=None)
    if rec.wait_cycles >= max(rec.sync_cycles, rec.compute_cycles_derived, rec.idle_cycles) and rec.wait_cycles > 0:
        return "inside primitive wait: waiting for peer data, recv flags, or downstream FIFO credit"
    if rec.idle_cycles >= max(rec.wait_cycles, rec.sync_cycles, rec.compute_cycles_derived) and largest_gap is not None:
        return f"outside primitive gap: {largest_gap.reason_hint}"
    if rec.sync_cycles >= max(rec.wait_cycles, rec.compute_cycles_derived, rec.idle_cycles) and rec.sync_cycles > 0:
        return "inside primitive sync: barrier, postPeer/postSend/postRecv, fence, or step bookkeeping"
    if rec.compute_cycles_derived > 0:
        return "mostly compute/copy inside primitive"
    return "no dominant cause identified"


def build_rows(records: List[TBRecord]) -> Tuple[List[dict], List[dict], List[dict], List[dict], dict]:
    if not records:
        return [], [], [], [], {}

    tb_rows: List[dict] = []
    prim_rows: List[dict] = []
    trace_rows: List[dict] = []
    gap_rows: List[dict] = []

    call_start = min(r.start_clk for r in records if r.start_clk > 0)
    call_stop = max(r.stop_clk for r in records)
    call_span = max(call_stop - call_start, 0)
    total_tb_cycles = 0
    total_busy_cycles = 0
    total_wait_cycles = 0
    total_compute_cycles = 0
    total_sync_cycles = 0
    total_idle_cycles = 0
    total_trace_dropped = 0
    prim_totals: Dict[str, int] = defaultdict(int)

    for rec in sorted(records, key=lambda r: (r.channel, r.work)):
        gaps = compute_gaps(rec)
        largest_gap = max((g.dur for g in gaps), default=0)
        largest_gap_reason = ""
        if gaps:
            g = max(gaps, key=lambda x: x.dur)
            largest_gap_reason = g.reason_hint
        dominant_reason = classify_tb_wait(rec, gaps)

        tb_rows.append(
            {
                "source": rec.source,
                "source_base": os.path.basename(rec.source),
                "op_count": rec.op_count,
                "channel": rec.channel,
                "work": rec.work,
                "tb_start_clk": rec.start_clk,
                "tb_stop_clk": rec.stop_clk,
                "tb_cycles": rec.tb_cycles,
                "primitive_busy_cycles": rec.busy_cycles,
                "wait_cycles": rec.wait_cycles,
                "compute_cycles": rec.compute_cycles_derived,
                "sync_cycles": rec.sync_cycles,
                "idle_cycles": rec.idle_cycles,
                "wait_pct_tb": (100.0 * rec.wait_cycles / rec.tb_cycles) if rec.tb_cycles else 0.0,
                "compute_pct_tb": (100.0 * rec.compute_cycles_derived / rec.tb_cycles) if rec.tb_cycles else 0.0,
                "sync_pct_tb": (100.0 * rec.sync_cycles / rec.tb_cycles) if rec.tb_cycles else 0.0,
                "busy_pct": rec.busy_pct,
                "idle_pct": rec.idle_pct,
                "wait_pct_prim": (100.0 * rec.wait_cycles / rec.busy_cycles) if rec.busy_cycles else 0.0,
                "compute_pct_prim": (100.0 * rec.compute_cycles_derived / rec.busy_cycles) if rec.busy_cycles else 0.0,
                "sync_pct_prim": (100.0 * rec.sync_cycles / rec.busy_cycles) if rec.busy_cycles else 0.0,
                "trace_count": len(rec.traces),
                "trace_dropped": rec.trace_dropped,
                "trace_complete": 1 if rec.trace_complete else 0,
                "largest_gap_cycles": largest_gap,
                "largest_gap_reason": largest_gap_reason,
                "dominant_wait_reason": dominant_reason,
            }
        )

        total_tb_cycles += rec.tb_cycles
        total_busy_cycles += rec.busy_cycles
        total_wait_cycles += rec.wait_cycles
        total_compute_cycles += rec.compute_cycles_derived
        total_sync_cycles += rec.sync_cycles
        total_idle_cycles += rec.idle_cycles
        total_trace_dropped += rec.trace_dropped

        for primitive, st in sorted(rec.prims.items(), key=lambda kv: kv[1].cycles, reverse=True):
            prim_totals[primitive] += st.cycles
            prim_rows.append(
                {
                    "source": rec.source,
                    "source_base": os.path.basename(rec.source),
                    "op_count": rec.op_count,
                    "channel": rec.channel,
                    "work": rec.work,
                    "primitive": primitive,
                    "cycles": st.cycles,
                    "calls": st.calls,
                    "pct_tb": (100.0 * st.cycles / rec.tb_cycles) if rec.tb_cycles else 0.0,
                    "pct_prim_sum": (100.0 * st.cycles / rec.busy_cycles) if rec.busy_cycles else 0.0,
                }
            )

        for tr in rec.traces:
            trace_rows.append(
                {
                    "source": rec.source,
                    "source_base": os.path.basename(rec.source),
                    "op_count": rec.op_count,
                    "channel": rec.channel,
                    "work": rec.work,
                    "group": tr.group,
                    "primitive": tr.primitive,
                    "seq": tr.seq,
                    "trace_start": tr.start,
                    "trace_stop": tr.stop,
                    "trace_dur": tr.dur,
                    "start_off": tr.start - rec.start_clk,
                    "stop_off": tr.stop - rec.start_clk,
                }
            )

        for gap in gaps:
            gap_rows.append(
                {
                    "source": gap.source,
                    "source_base": os.path.basename(gap.source),
                    "op_count": gap.op_count,
                    "channel": gap.channel,
                    "work": gap.work,
                    "gap_kind": gap.gap_kind,
                    "gap_start": gap.start,
                    "gap_stop": gap.stop,
                    "gap_cycles": gap.dur,
                    "prev_primitive": gap.prev_primitive,
                    "next_primitive": gap.next_primitive,
                    "trace_complete": gap.trace_complete,
                    "reason_hint": gap.reason_hint,
                }
            )

    summary = {
        "source": records[0].source,
        "source_base": os.path.basename(records[0].source),
        "op_count": records[0].op_count,
        "channel_count": len({(r.channel) for r in records}),
        "tb_count": len(records),
        "call_start_clk": call_start,
        "call_stop_clk": call_stop,
        "call_span_cycles": call_span,
        "tb_cycles_sum": total_tb_cycles,
        "primitive_busy_cycles_sum": total_busy_cycles,
        "wait_cycles_sum": total_wait_cycles,
        "compute_cycles_sum": total_compute_cycles,
        "sync_cycles_sum": total_sync_cycles,
        "idle_cycles_sum": total_idle_cycles,
        "busy_pct_vs_tb_sum": (100.0 * total_busy_cycles / total_tb_cycles) if total_tb_cycles else 0.0,
        "wait_pct_vs_tb_sum": (100.0 * total_wait_cycles / total_tb_cycles) if total_tb_cycles else 0.0,
        "compute_pct_vs_tb_sum": (100.0 * total_compute_cycles / total_tb_cycles) if total_tb_cycles else 0.0,
        "sync_pct_vs_tb_sum": (100.0 * total_sync_cycles / total_tb_cycles) if total_tb_cycles else 0.0,
        "idle_pct_vs_tb_sum": (100.0 * total_idle_cycles / total_tb_cycles) if total_tb_cycles else 0.0,
        "trace_dropped_sum": total_trace_dropped,
        "top_primitives": "|".join(p for p, _ in sorted(prim_totals.items(), key=lambda kv: kv[1], reverse=True)[:8]),
    }

    return tb_rows, prim_rows, trace_rows, gap_rows, summary


def write_csv(path: str, rows: List[dict], fieldnames: List[str]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def plot_call_timeline(outdir: str, summary: dict, records: List[TBRecord]) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
        import matplotlib.patches as mpatches  # type: ignore
    except Exception as e:
        print(f"[WARN] matplotlib not available, skip plot: {e}")
        return

    if not records:
        return

    cmap = plt.get_cmap("tab20")
    primitives = sorted({tr.primitive for rec in records for tr in rec.traces})
    color_of = {p: cmap(i % 20) for i, p in enumerate(primitives)}

    ordered = sorted(records, key=lambda r: (r.channel, r.work))
    call_start = summary["call_start_clk"]
    call_stop = summary["call_stop_clk"]
    call_span = max(call_stop - call_start, 1)

    fig_h = max(4.5, 0.45 * len(ordered) + 2.0)
    fig, ax = plt.subplots(figsize=(15, fig_h))

    for row, rec in enumerate(ordered):
        tb_offset = rec.start_clk - call_start
        ax.broken_barh([(tb_offset, rec.tb_cycles)], (row - 0.38, 0.76), facecolors="#e6e6e6", edgecolors="#b0b0b0", linewidth=0.5)

        groups = sorted({tr.group for tr in rec.traces}) or [0]
        lane_h = 0.72 / max(len(groups), 1)
        lane_of = {g: i for i, g in enumerate(groups)}
        for tr in rec.traces:
            lane = lane_of.get(tr.group, 0)
            y0 = row - 0.36 + lane * lane_h
            ax.broken_barh(
                [(tr.start - call_start, tr.dur)],
                (y0, lane_h * 0.92),
                facecolors=color_of[tr.primitive],
                edgecolors="black",
                linewidth=0.2,
            )

    ax.set_xlim(0, call_span)
    ax.set_xlabel("GPU clock offset within this NCCL call (ticks)")
    ax.set_ylabel("TB (channel/work)")
    ax.set_yticks(list(range(len(ordered))))
    ax.set_yticklabels([f"ch{rec.channel}/w{rec.work}" for rec in ordered])
    ax.set_title(f"TB lifecycle and primitive timeline ({summary['source_base']}, op_count={summary['op_count']})")
    ax.grid(axis="x", linestyle=":", alpha=0.35)

    handles = [mpatches.Patch(color="#e6e6e6", label="TB lifecycle")] + [
        mpatches.Patch(color=color_of[p], label=p) for p in primitives[:20]
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=8, ncol=1)

    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"tb_lifecycle_timeline_{summary['source_base']}_op{summary['op_count']}.png"), dpi=170)
    plt.close(fig)


def write_summary_text(path: str, summary: dict, tb_rows: List[dict], gap_rows: List[dict]) -> None:
    worst_tb = max(tb_rows, key=lambda x: x["idle_cycles"], default=None)
    worst_wait_tb = max(tb_rows, key=lambda x: x["wait_cycles"], default=None)
    worst_gap = max(gap_rows, key=lambda x: x["gap_cycles"], default=None)
    with open(path, "w") as f:
        for key, value in summary.items():
            f.write(f"{key}: {value}\n")
        if worst_tb is not None:
            f.write("\nworst_tb_by_idle:\n")
            for key in ["channel", "work", "tb_cycles", "primitive_busy_cycles", "idle_cycles", "idle_pct", "largest_gap_cycles", "largest_gap_reason", "trace_complete", "trace_dropped"]:
                f.write(f"{key}: {worst_tb.get(key)}\n")
        if worst_wait_tb is not None:
            f.write("\nworst_tb_by_internal_wait:\n")
            for key in ["channel", "work", "tb_cycles", "primitive_busy_cycles", "wait_cycles", "wait_pct_tb", "wait_pct_prim", "sync_cycles", "compute_cycles", "dominant_wait_reason", "trace_complete", "trace_dropped"]:
                f.write(f"{key}: {worst_wait_tb.get(key)}\n")
        if worst_gap is not None:
            f.write("\nworst_gap:\n")
            for key in ["channel", "work", "gap_kind", "gap_cycles", "prev_primitive", "next_primitive", "trace_complete", "reason_hint"]:
                f.write(f"{key}: {worst_gap.get(key)}\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Focused NCCL primitive profile analyzer")
    parser.add_argument("--input", nargs="+", required=True, help="Input CSV files or glob patterns")
    parser.add_argument("--outdir", default="prim_profile_focus", help="Output directory")
    parser.add_argument("--op-count", nargs="+", type=_parse_int_auto, help="Only analyze selected op_count values")
    parser.add_argument("--top-calls", type=int, default=1, help="If --op-count is omitted, analyze the top N calls by TB count")
    parser.add_argument("--no-plots", action="store_true", help="Skip PNG plot generation")
    args = parser.parse_args()

    files = resolve_inputs(args.input)
    if not files:
        print("[ERROR] no input files found", file=sys.stderr)
        return 2

    requested = set(args.op_count) if args.op_count else None
    records_map = load_records(files, requested)
    if not records_map:
        print("[ERROR] no records found", file=sys.stderr)
        return 2

    selected_calls = pick_default_calls(records_map.values(), requested, args.top_calls)
    if not selected_calls:
        print("[ERROR] no calls selected", file=sys.stderr)
        return 2

    os.makedirs(args.outdir, exist_ok=True)

    for op_count in selected_calls:
        call_records = [rec for rec in records_map.values() if rec.op_count == op_count]
        if not call_records:
            continue
        tb_rows, prim_rows, trace_rows, gap_rows, summary = build_rows(call_records)
        call_dir = os.path.join(args.outdir, f"op_{op_count}")
        os.makedirs(call_dir, exist_ok=True)

        write_csv(
            os.path.join(call_dir, "tb_summary.csv"),
            tb_rows,
            [
                "source", "source_base", "op_count", "channel", "work", "tb_start_clk", "tb_stop_clk", "tb_cycles",
                "primitive_busy_cycles", "wait_cycles", "compute_cycles", "sync_cycles", "idle_cycles",
                "wait_pct_tb", "compute_pct_tb", "sync_pct_tb", "busy_pct", "idle_pct",
                "wait_pct_prim", "compute_pct_prim", "sync_pct_prim",
                "trace_count", "trace_dropped", "trace_complete",
                "largest_gap_cycles", "largest_gap_reason", "dominant_wait_reason",
            ],
        )
        write_csv(
            os.path.join(call_dir, "tb_primitives.csv"),
            prim_rows,
            [
                "source", "source_base", "op_count", "channel", "work", "primitive", "cycles", "calls", "pct_tb", "pct_prim_sum",
            ],
        )
        write_csv(
            os.path.join(call_dir, "tb_trace.csv"),
            trace_rows,
            [
                "source", "source_base", "op_count", "channel", "work", "group", "primitive", "seq", "trace_start", "trace_stop", "trace_dur", "start_off", "stop_off",
            ],
        )
        write_csv(
            os.path.join(call_dir, "tb_gaps.csv"),
            gap_rows,
            [
                "source", "source_base", "op_count", "channel", "work", "gap_kind", "gap_start", "gap_stop", "gap_cycles",
                "prev_primitive", "next_primitive", "trace_complete", "reason_hint",
            ],
        )
        write_csv(
            os.path.join(call_dir, "call_summary.csv"),
            [summary],
            [
                "source", "source_base", "op_count", "channel_count", "tb_count", "call_start_clk", "call_stop_clk", "call_span_cycles",
                "tb_cycles_sum", "primitive_busy_cycles_sum", "wait_cycles_sum", "compute_cycles_sum", "sync_cycles_sum", "idle_cycles_sum",
                "busy_pct_vs_tb_sum", "wait_pct_vs_tb_sum", "compute_pct_vs_tb_sum", "sync_pct_vs_tb_sum", "idle_pct_vs_tb_sum",
                "trace_dropped_sum", "top_primitives",
            ],
        )
        write_summary_text(os.path.join(call_dir, "summary.txt"), summary, tb_rows, gap_rows)
        if not args.no_plots:
            plot_call_timeline(call_dir, summary, call_records)

        print(f"[OK] op_count={op_count} -> {os.path.abspath(call_dir)}")
        print(f"[OK]   TBs={summary['tb_count']} channels={summary['channel_count']} call_span={summary['call_span_cycles']}")
        print(
            f"[OK]   primitive_busy_sum={summary['primitive_busy_cycles_sum']} "
            f"wait_sum={summary['wait_cycles_sum']} compute_sum={summary['compute_cycles_sum']} "
            f"sync_sum={summary['sync_cycles_sum']} idle_sum={summary['idle_cycles_sum']}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
