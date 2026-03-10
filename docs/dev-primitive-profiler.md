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
type,channel,work,tb_cycles,prim_cycles_total,prim,cycles,calls,pct_tb,pct_prim_sum,start_clk,stop_clk
```

Row types:

- `type=tb`: one summary row per work item
- `type=prim`: one row per primitive with non-zero cycles for that work item

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

### 2) NCCL profile logs

When `NCCL_DEBUG=INFO` and `NCCL_DEBUG_SUBSYS=PROFILE` are enabled, profiler also prints `PRIMPROF` lines with the same information.

## Quick Analysis Examples

Aggregate total primitive cycles from CSV:

```bash
awk -F, 'NR>1 && $1=="prim" {sum[$6]+=$7} END {for (k in sum) print k, sum[k]}' /tmp/nccl_prim_profile.csv | sort -k2nr
```

Compute primitive percentage over all primitive cycles:

```bash
awk -F, '
NR>1 && $1=="prim" {sum[$6]+=$7; total+=$7}
END {for (k in sum) printf "%s,%.4f%%\n", k, 100.0*sum[k]/total}
' /tmp/nccl_prim_profile.csv | sort -t, -k2nr
```

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

