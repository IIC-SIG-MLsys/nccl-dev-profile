MPI_HOME=/usr/lib/x86_64-linux-gnu/openmpi/
NCCL_HOME=/home/xxxxxx/nccl/build
CUDA_HOME=/usr/local/cuda

export LD_LIBRARY_PATH=/home/xxxxx/nccl/build/lib:$LD_LIBRARY_PATH

bash run_all_reduce_perf_profile.sh

python3 ./nccl_prim_profile_report.py \
  --input "./nccl_prim_profile.csv" \
  --outdir ./nccl_prim_report \
  --topk 200