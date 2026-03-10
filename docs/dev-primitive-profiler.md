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
- `summary_operators.csv`
- `summary_channels.csv`
- `summary_files.csv`
- PNG plots (if `matplotlib` is installed):
  - `tb_busy_ratio_hist.png`
  - `channel_busy_waste_stacked.png`
  - `channel_waste_pct.png`
  - `tb_topk_waste_stacked.png`
  - `operator_busy_contrib_topk.png`
  - `tb_timeline_<source>.png` (TB-level primitive timeline / ordering)

Key TB-level columns in `tb_breakdown.csv`:

- `tb_cycles`: full TB lifecycle cycles
- `busy_cycles`: sum of primitive cycles on that TB
- `waste_cycles`: `max(tb_cycles - busy_cycles, 0)`
- `waste_pct`: waste ratio in TB lifecycle
- `operators`: operators seen on this TB

This makes it easy to inspect waste directly by sorting:

```bash
awk -F, 'NR==1{print;next}{print | "sort -t, -k7,7nr"}' /tmp/nccl_prim_report/tb_breakdown.csv | head
```

Inspect per-TB primitive ordering and occupancy:

```bash
# list the most wasteful TBs first
{
  head -n 1 /tmp/nccl_prim_report/tb_breakdown.csv
  tail -n +2 /tmp/nccl_prim_report/tb_breakdown.csv | sort -t, -k7,7nr
} | cut -d, -f2,3,4,5,6,7,9 | head -n 20

# inspect one TB's primitive sequence
awk -F, '$2==0 && $3==123 {print $0}' /tmp/nccl_prim_report/tb_trace_rows.csv | sort -t, -k9,9n
```

Interpretation hints:

- For one TB (`channel`,`work` fixed), compare `trace_start`/`trace_stop` across rows to see primitive order and gaps.
- TB utilization = `sum(trace_dur) / tb_cycles`.
- TB waste = `tb_cycles - sum(trace_dur)`.
- `tb_timeline_<source>.png` stacks many TBs together so you can see who starts earlier/later and where gaps are larger.
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
