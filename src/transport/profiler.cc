/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/
#include "transport.h"
#include "proxy.h"
#include "profiler.h"
#include "device.h"
#include "debug.h"
#include "param.h"
#include <cstdio>
#include <mutex>

static const char* ncclPrimProfileName[ncclPrimN] = {
  "send",
  "sendFromOutput",
  "directSend",
  "directSendFromOutput",
  "recv",
  "directRecv",
  "directRecvCopy",
  "copySend",
  "directCopySend",
  "recvSend",
  "recvCopySend",
  "directRecvCopyDirectSend",
  "directRecvDirectSend",
  "recvDirectSend",
  "directRecvSend",
  "recvCopyDirectSend",
  "recvReduceCopy",
  "directRecvReduceCopy",
  "recvReduceSend",
  "directRecvReduceSend",
  "recvReduceDirectSend",
  "directRecvReduceDirectSend",
  "recvReduceCopySend",
  "recvReduceCopyDirectSend",
  "directRecvReduceCopyDirectSend",
};

static std::mutex ncclPrimProfileFileMutex;
static FILE* ncclPrimProfileFile = nullptr;
static bool ncclPrimProfileFileInit = false;

static FILE* ncclPrimProfileGetFile() {
  std::lock_guard<std::mutex> lock(ncclPrimProfileFileMutex);
  if (!ncclPrimProfileFileInit) {
    ncclPrimProfileFileInit = true;
    const char* filePath = ncclGetEnv("NCCL_PRIM_PROFILE_FILE");
    if (filePath != nullptr && filePath[0] != '\0') {
      ncclPrimProfileFile = fopen(filePath, "a");
      if (ncclPrimProfileFile == nullptr) {
        WARN("Could not open NCCL_PRIM_PROFILE_FILE=%s", filePath);
      } else {
        fseek(ncclPrimProfileFile, 0, SEEK_END);
        long fileSize = ftell(ncclPrimProfileFile);
        if (fileSize == 0) {
          fprintf(ncclPrimProfileFile, "type,channel,work,tb_cycles,prim_cycles_total,prim,cycles,calls,pct_tb,pct_prim_sum,start_clk,stop_clk\n");
        }
        fflush(ncclPrimProfileFile);
      }
    }
  }
  return ncclPrimProfileFile;
}

static inline void ncclProfilerLogPrimSummary(const struct ncclProxySubArgs* sub, const struct ncclDevProfilerRecord* rec) {
  if (!ncclPrimProfileEnabled()) return;
  if (rec->counter < sub->base || rec->tbStop <= rec->tbStart) return;

  uint64_t tbCycles = rec->tbStop - rec->tbStart;
  uint64_t primCyclesTotal = 0;
  for (int p = 0; p < ncclPrimN; p++) primCyclesTotal += rec->primCycles[p];

  INFO(NCCL_PROFILE,
       "PRIMPROF channel %d work %llu tb_cycles %llu start_clk %llu stop_clk %llu prim_cycles_total %llu",
       sub->channelId,
       (unsigned long long)sub->base,
       (unsigned long long)tbCycles,
       (unsigned long long)rec->tbStart,
       (unsigned long long)rec->tbStop,
       (unsigned long long)primCyclesTotal);

  FILE* f = ncclPrimProfileGetFile();
  if (f != nullptr) {
    std::lock_guard<std::mutex> lock(ncclPrimProfileFileMutex);
    fprintf(f, "tb,%d,%llu,%llu,%llu,,0,0,0.00,0.00,%llu,%llu\n",
            sub->channelId,
            (unsigned long long)sub->base,
            (unsigned long long)tbCycles,
            (unsigned long long)primCyclesTotal,
            (unsigned long long)rec->tbStart,
            (unsigned long long)rec->tbStop);
  }

  for (int p = 0; p < ncclPrimN; p++) {
    if (rec->primCycles[p] == 0) continue;
    double pctTb = 100.0 * (double)rec->primCycles[p] / (double)tbCycles;
    double pctPrim = (primCyclesTotal == 0) ? 0.0 : (100.0 * (double)rec->primCycles[p] / (double)primCyclesTotal);
    INFO(NCCL_PROFILE,
         "PRIMPROF channel %d work %llu prim %s cycles %llu calls %u pct_tb %.2f pct_prim_sum %.2f",
         sub->channelId,
         (unsigned long long)sub->base,
         ncclPrimProfileName[p],
         (unsigned long long)rec->primCycles[p],
         rec->primCalls[p],
         pctTb,
         pctPrim);

    if (f != nullptr) {
      std::lock_guard<std::mutex> lock(ncclPrimProfileFileMutex);
      fprintf(f, "prim,%d,%llu,%llu,%llu,%s,%llu,%u,%.6f,%.6f,%llu,%llu\n",
              sub->channelId,
              (unsigned long long)sub->base,
              (unsigned long long)tbCycles,
              (unsigned long long)primCyclesTotal,
              ncclPrimProfileName[p],
              (unsigned long long)rec->primCycles[p],
              rec->primCalls[p],
              pctTb,
              pctPrim,
              (unsigned long long)rec->tbStart,
              (unsigned long long)rec->tbStop);
    }
  }
  if (f != nullptr) {
    std::lock_guard<std::mutex> lock(ncclPrimProfileFileMutex);
    fflush(f);
  }
}

static ncclResult_t profilerProxyConnect(struct ncclProxyConnection* connection, struct ncclProxyState* proxyState, void* reqBuff, int reqSize, void* respBuff, int respSize, int* done) {
  connection->proxyAppendPtr = &connection->proxyAppend;
  connection->shared = 0;
  return ncclSuccess;
}

// The following ncclProxySubArgs are overloaded by the profiler progress function:
// - base       : is set to the current value of workCounter[channelId]
// - posted     : is set to sub->nsteps to indicate that the profiler has started the event
// - transmitted: is set to sub->nsteps to indicate that the profiler has stopped the event
static ncclResult_t profilerProxyProgress(struct ncclProxyState* proxyState, struct ncclProxyArgs* args) {
  if (args->state == ncclProxyOpReady) {
    for (int s = 0; s < args->nsubs; s++) {
      struct ncclProxySubArgs* sub = args->subs + s;
      sub->base = sub->workCounter;
      sub->posted = sub->transmitted = 0;
    }
    args->state = ncclProxyOpProgress;
  }
  if (args->state == ncclProxyOpProgress) {
    for (int s = 0; s < args->nsubs; s++) {
      struct ncclProxySubArgs* sub = args->subs + s;
      struct ncclDevProfiler* workStarted = (struct ncclDevProfiler *)sub->sendbuff;
      struct ncclDevProfiler* workCompleted = (struct ncclDevProfiler *)sub->recvbuff;
      if (sub->posted < sub->nsteps && sub->base <= workStarted[sub->channelId].data[sub->base%MAX_PROFILER_EVENTS_PER_CHANNEL].counter) {
        ncclProfilerStartKernelChEvent(args, s, workStarted[sub->channelId].data[sub->base%MAX_PROFILER_EVENTS_PER_CHANNEL].timestamp);
        sub->posted = sub->nsteps;
        continue; // allow events on every channel to start
      }
      if (sub->transmitted < sub->nsteps && sub->base <= workCompleted[sub->channelId].data[sub->base%MAX_PROFILER_EVENTS_PER_CHANNEL].counter) {
        ncclProfilerLogPrimSummary(sub, &workCompleted[sub->channelId].data[sub->base%MAX_PROFILER_EVENTS_PER_CHANNEL]);
        ncclProfilerStopKernelChEvent(args, s, workCompleted[sub->channelId].data[sub->base%MAX_PROFILER_EVENTS_PER_CHANNEL].timestamp);
        sub->transmitted = sub->nsteps;
        args->done++;
      }
    }
    if (args->done == args->nsubs) args->state = ncclProxyOpNone;
  }
  return ncclSuccess;
}

struct ncclTransport profilerTransport = {
  "Prof",
  NULL,
  { NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL },
  { NULL, NULL, NULL, NULL, NULL, profilerProxyConnect, NULL, profilerProxyProgress, NULL, NULL }
};
