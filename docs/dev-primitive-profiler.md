# Dev Primitive Profiler

This document describes the in-tree device-side primitive profiler added for NCCL runtime debugging and performance analysis.

## What It Measures

The profiler records per-work (per channel work counter) timing on GPU and exports primitive-level breakdown:

- TB lifecycle:
  - `tbStart`: timestamp captured before running one work item in kernel
  - `tbStop`: timestamp captured after that work item finishes
- Primitive breakdown (inside `ProtoSimple` primitives):
  - total cycles per primitive (`primCycles`)
  - call count per primitive (`primCalls`)

All timestamps/cycles come from GPU `globaltimer` ticks.

## Enable Profiling

Set:

```bash
export NCCL_PRIM_PROFILE=1
```

Optional (recommended for offline analysis):

```bash
export NCCL_PRIM_PROFILE_FILE=/tmp/nccl_prim_profile.csv
```

Optional (console logs):

```bash
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=PROFILE
```

## Run Example

```bash
NCCL_PRIM_PROFILE=1 \
NCCL_PRIM_PROFILE_FILE=/tmp/nccl_prim_profile.csv \
NCCL_DEBUG=INFO \
NCCL_DEBUG_SUBSYS=PROFILE \
./your_nccl_program
```

## Output Formats

### 1) CSV file (`NCCL_PRIM_PROFILE_FILE`)

When `NCCL_PRIM_PROFILE_FILE` is set, profiler appends CSV rows:

```text
type,channel,work,tb_cycles,prim_cycles_total,prim,cycles,calls,pct_tb,pct_prim_sum,start_clk,stop_clk,trace_seq,trace_start,trace_stop,trace_dur,trace_start_off,trace_stop_off,trace_dropped
```

Row types:

- `type=tb`: one summary row per work item
- `type=prim`: one row per primitive with non-zero cycles for that work item
- `type=trace`: one row per primitive invocation, preserving in-TB order and timing

Column meanings:

- `channel`: channel id
- `work`: work counter id
- `tb_cycles`: `tbStop - tbStart`
- `prim_cycles_total`: sum of all primitive cycles in this work item
- `prim`: primitive name (`send`, `directRecvReduceDirectSend`, ...)
- `cycles`: cycles consumed by this primitive
- `calls`: number of calls for this primitive
- `pct_tb`: `cycles / tb_cycles * 100`
- `pct_prim_sum`: `cycles / prim_cycles_total * 100`
- `start_clk`, `stop_clk`: raw GPU timer values
- `trace_seq`: primitive call order within TB
- `trace_start`, `trace_stop`: raw timer for one primitive call
- `trace_dur`: `trace_stop - trace_start`
- `trace_start_off`, `trace_stop_off`: offset from TB start clock
- `trace_dropped`: number of trace events dropped because `NCCL_PRIM_TRACE_MAX_PER_WORK` was exceeded

### 2) NCCL profile logs

When `NCCL_DEBUG=INFO` and `NCCL_DEBUG_SUBSYS=PROFILE` are enabled, profiler also prints `PRIMPROF` lines with the same information.

## Quick Analysis Examples

Use the built-in post-processing script:

```bash
python3 tools/nccl_prim_profile_report.py \
  --input /tmp/nccl_prim_profile_rank*.csv \
  --outdir /tmp/nccl_prim_report
```

Outputs include:

- `summary_global.txt`
- `tb_breakdown.csv` (one row per TB/work item)
- `tb_operator_rows.csv` (one row per TB+operator)
- `tb_trace_rows.csv` (one row per primitive call; includes in-TB order and timing offsets)
- `channel_breakdown.csv` (one row per `source+channel`, all works merged)
- `channel_operator_rows.csv` (one row per `source+channel+operator`)
- `summary_operators.csv`
- `summary_channels.csv` (same content as `channel_breakdown.csv` for compatibility)
- `summary_files.csv`
- PNG plots (if `matplotlib` is installed):
  - `tb_busy_ratio_hist.png`
  - `channel_busy_waste_stacked.png`
  - `channel_waste_pct.png`
  - `channel_topk_waste_stacked.png`
  - `operator_busy_contrib_topk.png`
  - `channel_timeline_<source>.png` (each row is one channel, all works merged on one axis)

Key TB-level columns in `tb_breakdown.csv`:

- `tb_cycles`: full TB lifecycle cycles
- `busy_cycles`: sum of primitive cycles on that TB
- `waste_cycles`: `max(tb_cycles - busy_cycles, 0)`
- `waste_pct`: waste ratio in TB lifecycle
- `operators`: operators seen on this TB

Channel-merged lifecycle columns (in `channel_breakdown.csv`):

- `channel_cycles`: channel lifecycle span (`max(stop_clk)-min(start_clk)` over all works in that channel)
- `busy_cycles`: union of trace durations on the channel timeline
- `waste_cycles`: `max(channel_cycles - busy_cycles, 0)`
- `work_count`: number of different `work` ids merged into this channel

This makes it easy to inspect waste directly by channel:

```bash
{
  head -n 1 /tmp/nccl_prim_report/channel_breakdown.csv
  tail -n +2 /tmp/nccl_prim_report/channel_breakdown.csv | sort -t, -k12,12nr
} | head
```

Inspect primitive ordering on one channel timeline:

```bash
# inspect one channel (all works mixed) ordered by absolute start time
awk -F, '$3==0 {print $0}' /tmp/nccl_prim_report/tb_trace_rows.csv | sort -t, -k10,10n
```

Interpretation hints:

- `channel_timeline_<source>.png` puts all works of the same channel on the same row, so you can directly see cross-work ordering and idle gaps.
- Channel utilization = `busy_cycles / channel_cycles` (from `channel_breakdown.csv`).
- Channel waste = `channel_cycles - busy_cycles`.
- `busy_cycles_sum` may be larger than `busy_cycles` if trace windows overlap; overlap amount is in `oversub_cycles`.
- Across different GPUs/ranks, absolute clocks may not be globally synchronized; compare ordering and shape primarily within one source file.

## Notes and Limitations

- Current instrumentation is in `prims_simple.h` (`ProtoSimple` path).
- Data is collected from thread `0` per block to reduce profiling overhead.
- Units are GPU timer ticks (not directly microseconds).
- `MAX_PROFILER_EVENTS_PER_CHANNEL` ring size is 64; very heavy backlog can overwrite old slots.
- For multi-process runs, use per-rank files to avoid inter-process append contention, for example:
  - `/tmp/nccl_prim_profile_rank${RANK}.csv`

## Related Source Files

- `src/include/device.h`
- `src/device/common.h`
- `src/device/prims_simple.h`
- `src/transport/profiler.cc`
- `src/plugin/profiler.cc`
