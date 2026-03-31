#!/usr/bin/env bash
set -euo pipefail

NCCL_BIN="/home/xxxxxxx/nccl-tests/build/all_reduce_perf"
# NCCL_BIN="/home/liujinyao/IIC/nccl-tests/build/alltoall_perf"
NUM_RANKS=8
BIND_OPTS="-bind-to none -map-by slot"
NCCL_PROFILE_FILE="/home/xxxxxx/nccl_prim_profile.csv"
NCCL_LIB_DIR="/home/xxxxxx/nccl/build/lib"

rm -f "$NCCL_PROFILE_FILE"

mpirun -np "$NUM_RANKS" $BIND_OPTS bash -c '
RANK=${OMPI_COMM_WORLD_RANK:-0}

unset CUDA_VISIBLE_DEVICES
unset DISPLAY  # X11 authorization

export LD_LIBRARY_PATH='"$NCCL_LIB_DIR"':$LD_LIBRARY_PATH

if [ "$RANK" -eq 0 ]; then
  echo "[INFO] Rank 0: profiling enabled"
  export NCCL_PRIM_PROFILE=1
  export NCCL_PRIM_PROFILE_FILE='"$NCCL_PROFILE_FILE"'
  export NCCL_DEBUG=INFO
  export NCCL_DEBUG_SUBSYS=PROFILE
else
  echo "[INFO] Rank $RANK: normal run"
fi

export NCCL_MAX_NCHANNELS=4
export NCCL_MIN_NCHANNELS=4

exec '"$NCCL_BIN"' -b 2G -e 2G -f 2 -g 1 -w 0 -n 1 -c 0
'
