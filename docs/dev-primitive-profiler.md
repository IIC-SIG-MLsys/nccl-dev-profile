# Dev Primitive Profiler

这个 profiler 现在只围绕一件事：

- 一次 NCCL 调用里，启动了哪些 TB
- 每个 TB 生命周期多长
- TB 里各个 primitive 占了多少时间
- TB 有没有空闲
- 空闲发生在什么位置，可能是在等什么

## 采集内容

设备侧会记录：

- `tbStart` / `tbStop`
- 每种 primitive 的总 cycles / calls
- 每次 primitive 调用的 trace
  - `trace_group`: 同一个 TB 里的 primitives group
  - `trace_seq`: trace 顺序
  - `trace_start` / `trace_stop`

Host 侧落盘时还会带上：

- `op_count`: 这次 NCCL 调用的 id
- `channel`
- `work`

所以现在可以从一个混合 CSV 里直接切出“某一次 NCCL 调用”。

## CSV 格式

`NCCL_PRIM_PROFILE_FILE` 输出列：

```text
type,op_count,channel,work,tb_cycles,prim_cycles_total,prim,cycles,calls,pct_tb,pct_prim_sum,start_clk,stop_clk,trace_group,trace_seq,trace_start,trace_stop,trace_dur,trace_start_off,trace_stop_off,trace_dropped
```

含义：

- `type=tb`: 一个 TB/work 的汇总
- `type=prim`: 这个 TB 上某种 primitive 的累计时间
- `type=trace`: 这个 TB 上一次 primitive 调用

关键字段：

- `op_count`: 一次 NCCL 调用
- `channel + work`: 一个 TB/work
- `tb_cycles`: TB 整个生命周期
- `prim_cycles_total`: TB 上所有 primitive 的总时间
- `trace_group`: TB 内不同 primitives group
- `trace_dropped`: trace 是否被截断

## 分析脚本

脚本：

- [tools/nccl_prim_profile_report.py](/Users/jacelau/code/opencode/nccl/tools/nccl_prim_profile_report.py)

运行：

```bash
python3 tools/nccl_prim_profile_report.py \
  --input /tmp/nccl_prim_profile_rank*.csv \
  --outdir /tmp/nccl_prim_focus
```

如果只看一个调用：

```bash
python3 tools/nccl_prim_profile_report.py \
  --input /tmp/nccl_prim_profile_rank*.csv \
  --op-count 42 \
  --outdir /tmp/nccl_prim_focus_call42
```

如果不指定 `--op-count`，默认只分析 TB 数最多的那个调用。

## 输出文件

每个调用一个目录，例如：`/tmp/nccl_prim_focus/op_42/`

里面只有这几个文件：

- `call_summary.csv`
  - 一行，就是这次 NCCL 调用的整体情况
- `tb_summary.csv`
  - 一行一个 TB/work
  - 重点看：`tb_cycles`、`primitive_busy_cycles`、`idle_cycles`、`idle_pct`
- `tb_primitives.csv`
  - 一个 TB 上各 primitive 的累计时间分布
- `tb_trace.csv`
  - 一个 TB 上每次 primitive 调用的时序
- `tb_gaps.csv`
  - TB 生命周期里不在 primitive 内部的 gap
  - 用来定位“空闲发生在什么位置”
- `summary.txt`
  - 直接给出最空闲的 TB 和最大的 gap
- `tb_lifecycle_timeline_<source>_op<op_count>.png`
  - 灰底是 TB 生命周期
  - 彩条是 primitive trace
  - 空白就是没有落在 primitive 内的 gap

## 怎么看

### 1. 先看整次调用

看 `call_summary.csv`：

- `call_span_cycles`: 这次 NCCL 调用从最早 TB 开始到最晚 TB 结束的跨度
- `tb_cycles_sum`: 所有 TB 生命周期求和
- `primitive_busy_cycles_sum`: 所有 TB 上 primitive 时间求和
- `idle_cycles_sum`: 所有 TB 上非-primitive 时间求和

### 2. 找最空闲的 TB

看 `tb_summary.csv`：

- `idle_cycles = tb_cycles - primitive_busy_cycles`
- `idle_pct` 越大，说明这个 TB 生命周期里非-primitive 部分越多
- `largest_gap_cycles` 是这个 TB 上最大的空档
- `largest_gap_reason` 是源码语义下的原因提示

### 3. 看 primitive 分布

看 `tb_primitives.csv`：

- `pct_tb`: 某个 primitive 占这个 TB 生命周期的比例
- `pct_prim_sum`: 某个 primitive 占所有 primitive 时间的比例

### 4. 看空闲发生在什么位置

看 `tb_gaps.csv`：

- `before_first_primitive`
  - 通常是 primitive setup、connector sync、pointer exchange、barrier
- `between_primitives`
  - 通常是 primitive 边界上的 bookkeeping / barrier / postPeer / 下一段 setup
  - 真正的 peer/data wait 通常已经算在 primitive 里面了
- `after_last_primitive`
  - 通常是 teardown、final sync、析构等待、kernel epilogue

如果 `trace_complete=0`，说明 trace 被截断了，gap 只能作为提示，不能当完整事实。

### 5. 看图

`tb_lifecycle_timeline_*.png`：

- 每一行一个 TB：`chX/wY`
- 灰色整条：TB 生命周期
- 彩色块：primitive
- 空白：非-primitive gap

这是现在最核心的图。

## 注意事项

- 现在覆盖 `ProtoSimple` / `ProtoLL` / `ProtoLL128`
- multi-group TB 已经支持，trace 里会带 `trace_group`
- `NCCL_PRIM_TRACE_MAX_PER_WORK=1024`
- 如果 `trace_dropped > 0`，说明这个 TB 的 trace 仍然不完整
- GPU `globaltimer` 是 ticks，不是 us
- 不同 rank/GPU 的绝对时钟不能直接横向对齐，优先看同一个 source 文件内的相对关系

## 相关源码

- [src/include/device.h](/Users/jacelau/code/opencode/nccl/src/include/device.h)
- [src/device/common.h](/Users/jacelau/code/opencode/nccl/src/device/common.h)
- [src/device/prims_simple.h](/Users/jacelau/code/opencode/nccl/src/device/prims_simple.h)
- [src/device/prims_ll.h](/Users/jacelau/code/opencode/nccl/src/device/prims_ll.h)
- [src/device/prims_ll128.h](/Users/jacelau/code/opencode/nccl/src/device/prims_ll128.h)
- [src/transport/profiler.cc](/Users/jacelau/code/opencode/nccl/src/transport/profiler.cc)
