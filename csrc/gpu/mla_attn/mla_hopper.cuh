/*
 * Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri
 * Dao. Licensed under the BSD 3-Clause.
 *
 * Modified by the FlashInfer team.
 */
#ifndef FLASHINFER_ATTENTION_HOPPER_PREFILL_SM90_CUH_
#define FLASHINFER_ATTENTION_HOPPER_PREFILL_SM90_CUH_

#include <cuda.h>
#include <cuda_device_runtime_api.h>
#include <cutlass/arch/reg_reconfig.h>
#include <cutlass/array.h>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/numeric_types.h>

#include <type_traits>
#include <vector>

#include "cutlass_utils.cuh"
#include "attention_updater.cuh"
#include "mask.cuh"
#include "cute/tensor.hpp"
#include "cutlass/pipeline/pipeline.hpp"
#include "epilogue.cuh"
#include "kernel_traits.cuh"
#include "mainloop_mma.cuh"
#include "sparse_mainloop.cuh"
#include "utils.cuh"

// #define DEBUG_MLA

namespace flashinfer {

using namespace cute;

template <typename DTypeQ_, typename DTypeKV_, typename DTypeO_, typename IdType_>
struct Params {
    using DTypeQ = DTypeQ_;
    using DTypeKV = DTypeKV_;
    using DTypeO = DTypeO_;
    using IdType = IdType_;

    alignas(16) DTypeQ *Q; // [token_num, head_num, dim_head]
    alignas(16) DTypeKV *KV; // [max_block_num, block_size, dim_head]
    alignas(16) DTypeO *O; // [token_num, head_num, dim_head]
    alignas(16) DTypeO *O_tmp; // [num_chunks, bsz, head_num, dim_head]
    alignas(16) float *m; // [num_chunks, bsz * max_draft_token_num * head_num]
    alignas(16) float *d; // [num_chunks, bsz * max_draft_token_num * head_num]

    alignas(16) IdType *block_tables;
    alignas(16) IdType *seq_lens_this_time;
    alignas(16) IdType *seq_lens_encoder;
    alignas(16) IdType *seq_lens_decoder;
    alignas(16) IdType *cumsum_q_seqlens;
    alignas(16) IdType *padding_offsets;

    alignas(16) IdType *batch_ids;
    alignas(16) IdType *tile_ids_per_batch;
    alignas(16) IdType *num_blocks_x;


    uint32_t q_stride_bsz;
    uint32_t q_stride_head_num;

    uint32_t kv_stride_block_num;
    uint32_t kv_stride_block_size;

    uint32_t o_stride_bsz;
    uint32_t o_stride_head_num;

    int bsz;
    int token_num;
    int max_seq_len;
    int max_block_num;
    int max_block_num_per_seq;
    int q_num_head;
    int qk_head_dim;
    int vo_head_dim;
    int block_size;
    int max_draft_token_num;
    int chunk_size; // todo: find better chunk_size(can't control diff)
    int chunk_num;

    float sm_scale;
};

// #define DISPATCH_GQA_GROUP_SIZE(group_size, GROUP_SIZE, ...) \
//   if (group_size == 8) {                                     \
//     constexpr size_t GROUP_SIZE = 8;                         \
//     __VA_ARGS__                                              \
//   } else if (group_size == 16) {                             \
//     constexpr size_t GROUP_SIZE = 16;                        \
//     __VA_ARGS__                                              \
//   } else if (group_size == 64) {                             \
//     constexpr size_t GROUP_SIZE = 64;                        \
//     __VA_ARGS__                                              \
//   }

#define DISPATCH_GQA_GROUP_SIZE(group_size, GROUP_SIZE, ...) \
  if (group_size == 8) {                                     \
    constexpr size_t GROUP_SIZE = 8;                         \
    __VA_ARGS__                                              \
  } else if (group_size == 64) {                              \
    constexpr size_t GROUP_SIZE = 64;                         \
    __VA_ARGS__                                              \
  }

template <typename CollectiveMainloop, typename CollectiveEpilogue, typename Ktraits, bool CAUSAL, int SM_COUNT = 132>
__global__ void __launch_bounds__(Ktraits::NUM_WARPS * cutlass::NumThreadsPerWarp, 1)
MLAWithKVCacheKernel(CUTE_GRID_CONSTANT
                     typename CollectiveMainloop::Params const mainloop_params,
                     CUTE_GRID_CONSTANT
                     typename CollectiveEpilogue::Params const epilogue_params) {

  using DTypeQ = typename Ktraits::DTypeQ;
  using DTypeKV = typename Ktraits::DTypeKV;
  using DTypeO = typename Ktraits::DTypeO;
  using DTypeQKAccum = typename Ktraits::DTypeQKAccum;
  using TileShape_QKD = typename Ktraits::TileShape_QKD;
  using TileShape_PDV = typename Ktraits::TileShape_PDV;

  static constexpr int NUM_MMA_THREADS = Ktraits::NUM_MMA_THREADS;
  static constexpr int NUM_COPY_THREADS = Ktraits::NUM_PRODUCER_THREADS;
  static constexpr int CTA_Q = Ktraits::CTA_Q;
  static constexpr int CTA_KV = Ktraits::CTA_KV;
  const int num_blocks_x = mainloop_params.num_blocks_x[0];

  static constexpr bool use_tma_load_kv = CollectiveMainloop::USE_TMA_LOAD_KV;
#ifdef DEBUG_MLA
  if (thread(0)) {
    printf("use_tma_load_kv: %d\n", (int)use_tma_load_kv);
    printf("NUM_MMA_THREADS: %d\b", NUM_MMA_THREADS);
    printf("NUM_COPY_THREADS: %d\b", NUM_COPY_THREADS);
  }
  __syncthreads();
#endif

  using MainloopPipeline = typename CollectiveMainloop::MainloopPipeline;
  using PipelineParams = typename MainloopPipeline::Params;
  using PipelineState = typename MainloopPipeline::PipelineState;

  extern __shared__ char shared_memory[];
  auto& shared_storage = *reinterpret_cast<typename Ktraits::SharedStorage*>(shared_memory);

  int const lane_predicate = cute::elect_one_sync();
  int const warp_idx = cutlass::canonical_warp_idx_sync();

  // Obtain warp index
  int const warp_group_thread_idx = threadIdx.x % cutlass::NumThreadsPerWarpGroup;

  PipelineParams pipeline_params;
  int warp_group_idx = cutlass::canonical_warp_group_idx();
  pipeline_params.role = warp_group_idx == 0 ? MainloopPipeline::ThreadCategory::Producer
                                             : MainloopPipeline::ThreadCategory::Consumer;
  pipeline_params.producer_arv_count = NUM_COPY_THREADS;
  pipeline_params.consumer_arv_count = NUM_MMA_THREADS;
  MainloopPipeline pipeline_q(shared_storage.pipeline_q, pipeline_params);
  MainloopPipeline pipeline_kv(shared_storage.pipeline_kv, pipeline_params);
  __syncthreads();

  CollectiveMainloop collective_mainloop;
  CollectiveEpilogue collective_epilogue;
  
  if (warp_group_idx == 0) {
    // producer
    // cutlass::arch::warpgroup_reg_dealloc<80>();
    cutlass::arch::warpgroup_reg_dealloc<72>();
    const uint32_t warp_idx_in_warpgroup = __shfl_sync(0xffffffff, warp_idx % 4, 0);
    PipelineState smem_pipe_write_q = cutlass::make_producer_start_state<MainloopPipeline>();
    PipelineState smem_pipe_write_kv = cutlass::make_producer_start_state<MainloopPipeline>();
    for (int i = blockIdx.x; i < num_blocks_x; i += SM_COUNT) {
      const int bid = mainloop_params.batch_ids[i];
      const int tile_id = mainloop_params.tile_ids_per_batch[i];
      const int seq_len_now = mainloop_params.seq_lens_this_time[bid];
      const int seq_len_encoder_now = mainloop_params.seq_lens_encoder[bid];
      const int seq_len_decoder_now = mainloop_params.seq_lens_decoder[bid];
      const int start_token_idx = mainloop_params.cumsum_q_seqlens[bid];
#ifdef DEBUG_MLA
      if (block(0) && thread(0)) {
        printf("i: %d, bid: %d\n", i, bid);
        printf("bid: %d, tile_id: %d, seq_len_now: %d, seq_len_encoder_now: %d, seq_len_decoder_now: %d, start_token_idx: %d\n"
              , bid, tile_id, seq_len_now, seq_len_encoder_now, seq_len_decoder_now, start_token_idx);
      }
#endif
      // load Q
      collective_mainloop.load_q(
          mainloop_params,
          pipeline_q,
          smem_pipe_write_q,
          shared_storage);
#ifdef DEBUG_MLA
      if (block(0) && thread(0)) {
        printf("load q done\n");
      }
#endif
      
      // load kv
      collective_mainloop.load_kv(
          mainloop_params,
          pipeline_kv,
          smem_pipe_write_kv,
          shared_storage,
          bid,
          seq_len_decoder_now,
          tile_id
      );
#ifdef DEBUG_MLA
      if (block(0) && thread(0)) {
        printf("load kv done\n");
      }
#endif
    }
    collective_mainloop.load_tail(pipeline_q, smem_pipe_write_q, pipeline_kv, smem_pipe_write_kv);
  } else {
    // consumer
    // cutlass::arch::warpgroup_reg_alloc<208>(); // 384 threads max_reg_num = 168, 80 * 128 / 256 + 168 = 208
    cutlass::arch::warpgroup_reg_alloc<216>(); // 384 threads max_reg_num = 168, 80 * 128 / 256 + 168 = 208
    PipelineState smem_pipe_read_q;
    PipelineState smem_pipe_read_kv;

    typename Ktraits::TiledMmaPV tiled_mma_pv;
    Tensor tOrO = partition_fragment_C(tiled_mma_pv, select<0, 1>(TileShape_PDV{}));
#ifdef DEBUG_MLA
    if (thread(128)) {
        printf("\ntOtO: \n");
        print(tOrO);
    }
#endif
    clear(tOrO);

    auto attention_updater = OnlineSoftmax<2 * size<1>(tOrO), /*WITH_SCALE=*/true>(mainloop_params.sm_scale);
    
    for (int i = blockIdx.x; i < num_blocks_x; i += SM_COUNT) {
      const int bid = mainloop_params.batch_ids[i];
      const int tile_id = mainloop_params.tile_ids_per_batch[i];
      const int seq_len_now = mainloop_params.seq_lens_this_time[bid];
      const int seq_len_encoder_now = mainloop_params.seq_lens_encoder[bid];
      const int seq_len_decoder_now = mainloop_params.seq_lens_decoder[bid];
      const int start_token_idx = mainloop_params.cumsum_q_seqlens[bid];
      mma_f16<Ktraits, CAUSAL>(
          mainloop_params, 
          pipeline_q, 
          smem_pipe_read_q,
          pipeline_kv, 
          smem_pipe_read_kv,
          tOrO, 
          attention_updater, 
          threadIdx.x - NUM_COPY_THREADS,
          bid,
          seq_len_decoder_now,
          seq_len_now,
          tile_id,
          shared_storage);
      collective_epilogue.store(
          epilogue_params, 
          tOrO, 
          attention_updater.get_lse(),
          shared_storage,
          tiled_mma_pv, 
          threadIdx.x - NUM_COPY_THREADS,
          bid,
          mainloop_params.bsz,
          seq_len_now,
          start_token_idx,
          tile_id,
          seq_len_decoder_now,
          mainloop_params.chunk_size,
          mainloop_params.o_stride_bsz);
    }
    collective_epilogue.store_tail();
  }
}

__global__ void split_q_block(const int * __restrict__ seq_lens_q,
                              const int * __restrict__ seq_lens_encoder,
                              const int * __restrict__ seq_lens_decoder,
                              int * __restrict__ batch_ids,
                              int * __restrict__ tile_ids_per_batch,
                              int * __restrict__ num_blocks_x,
                              const int bsz,
                              const int num_rows_per_block,
                              const int chunk_size,
                              const int GROUP_SIZE,
                              const bool is_encoder) {
  if (threadIdx.x == 0) {
    int gridx = 0;
    int index = 0;
    for (uint32_t bid = 0; bid < bsz; bid++) {
      int seq_len = seq_lens_q[bid];
      int seq_len_encoder = seq_lens_encoder[bid];
      int seq_len_decoder = seq_lens_decoder[bid];

      if (seq_len == 0) continue;

      int loop_times;
      if (is_encoder) {
        loop_times = cute::ceil_div(seq_len * GROUP_SIZE, num_rows_per_block);
        if (seq_len_decoder > 0) {
          loop_times = 0;
        }
      } else {
        loop_times = cute::ceil_div(seq_len_decoder, chunk_size);
        if (seq_len_encoder > 0) {
          loop_times = 0;
        }
      }
      for (uint32_t tile_id = 0; tile_id < loop_times; tile_id++) {
        batch_ids[index] = bid;
        tile_ids_per_batch[index++] = tile_id;
      }
      gridx += loop_times;
    }
    *num_blocks_x = gridx;
  }
}

template <typename KernelTraits, bool CAUSAL, typename Params>
cudaError_t BatchMLAWithPagedKVCacheKernelTraitsDispatched(Params& params,
                                                           cudaStream_t stream) {
  using DTypeQ = typename KernelTraits::DTypeQ;
  using DTypeKV = typename KernelTraits::DTypeKV;
  using DTypeO = typename KernelTraits::DTypeO;
  using IdType = typename KernelTraits::IdType;

  using CollectiveMainloop =
      SparseCollectiveMainloop<KernelTraits, CAUSAL>;
  using CollectiveEpilogue = CollectiveEpilogue<KernelTraits>;

  split_q_block<<<1, 32, 0, stream>>>(
    params.seq_lens_this_time,
    params.seq_lens_encoder,
    params.seq_lens_decoder,
    params.batch_ids,
    params.tile_ids_per_batch,
    params.num_blocks_x,
    params.bsz,
    KernelTraits::CTA_Q,
    params.chunk_size,
    KernelTraits::GROUP_SIZE,
    false // is_encoder
  );
#ifdef DEBUG_MLA
  printf("chunk_num: :%d\n", params.chunk_num);
  printf("bsz: :%d\n", params.bsz);
  printf("q_num_head: :%d\n", params.q_num_head);
#endif
  typename CollectiveMainloop::Params mainloop_params = CollectiveMainloop::to_underlying_arguments({
      make_layout(make_shape(KernelTraits::CTA_Q, params.qk_head_dim), make_stride(params.qk_head_dim, _1{})), // layout q
      make_layout(make_shape(params.block_size, params.qk_head_dim, params.max_block_num), make_stride(params.qk_head_dim, _1{}, params.block_size * params.qk_head_dim)),
      make_layout(make_shape(params.chunk_num, params.bsz * params.max_draft_token_num * params.q_num_head), make_stride(params.bsz * params.max_draft_token_num * params.q_num_head, _1{})),
    //   make_layout(make_shape(KernelTraits::CTA_KV, params.qk_head_dim), make_stride(params.qk_head_dim, _1{})), // layout q
      params.Q,
      params.KV,
      params.m,
      params.d,
      params.block_tables,
      params.seq_lens_this_time,
      params.seq_lens_encoder,
      params.seq_lens_decoder,
      params.cumsum_q_seqlens,
      params.batch_ids,
      params.tile_ids_per_batch,
      params.num_blocks_x,
      params.sm_scale,
      params.bsz,
      params.max_block_num,
      params.max_block_num_per_seq,
      params.q_stride_bsz,
      params.q_stride_head_num,
      params.kv_stride_block_num,
      params.kv_stride_block_size,
      params.o_stride_bsz,
      params.o_stride_head_num,
      params.chunk_size,
      params.max_draft_token_num
  });
  typename CollectiveEpilogue::Params epilogue_params = CollectiveEpilogue::to_underlying_arguments_ntma({
      params.O,
      make_layout(make_shape(KernelTraits::CTA_Q, params.vo_head_dim), make_stride(params.vo_head_dim, _1{})), // layout O
      params.O_tmp,
      make_layout(make_shape(KernelTraits::CTA_Q, params.vo_head_dim), make_stride(params.vo_head_dim, _1{})) // layout O_tmp
  });

  // Get the ptr to kernel function.
  auto kernel =
      (void*)MLAWithKVCacheKernel<CollectiveMainloop, CollectiveEpilogue, KernelTraits, CAUSAL, 132>;
  int smem_size = sizeof(typename KernelTraits::SharedStorage);
#ifdef DEBUG_MLA
  printf("smem_size: %d KB\n", smem_size / 1024);
#endif
  FLASHINFER_CUDA_CALL(
      cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

  int device;
  cudaGetDevice(&device);
  int multiprocessor_count;
  FLASHINFER_CUDA_CALL(
      cudaDeviceGetAttribute(&multiprocessor_count, cudaDevAttrMultiProcessorCount, device));
#ifdef DEBUG_MLA
  printf("multiprocessor_count: %d\n", multiprocessor_count);
#endif
  int act_blocks_per_sm;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &act_blocks_per_sm, kernel, KernelTraits::NUM_WARPS * 32, smem_size);
#ifdef DEBUG_MLA
  printf("act_blocks_per_sm: %d\n", act_blocks_per_sm);
#endif
  cudaDeviceProp devProp;
  cudaGetDeviceProperties(&devProp, device);
  
  dim3 grid_dims = {multiprocessor_count, 1, 1}; // todo: split kv
  static constexpr int ctaSize = KernelTraits::NUM_WARPS * 32;
#ifdef DEBUG_MLA
  printf("ctaSize: %d\n", ctaSize);
#endif
  dim3 block_dims(ctaSize);
  MLAWithKVCacheKernel<CollectiveMainloop, CollectiveEpilogue, KernelTraits, CAUSAL, 132><<<grid_dims, block_dims, smem_size, stream>>>(
    mainloop_params, epilogue_params
  );

  constexpr int vec_size = 16 / sizeof(DTypeO);
  constexpr int merge_block_size = 256;
  constexpr int blockx = KernelTraits::HEAD_DIM_VO / vec_size;
  constexpr int blocky = (merge_block_size + blockx - 1) / blockx;
  dim3 grids_merge(min(multiprocessor_count, params.token_num), params.q_num_head); // 128k is too large
  dim3 blocks_merge(blockx, blocky);
  merge_multi_chunks_kernel<DTypeO, vec_size, blocky, KernelTraits::HEAD_DIM_VO><<<grids_merge, blocks_merge, 0, stream>>>(
    params.O_tmp,
    params.m,
    params.d,
    params.seq_lens_this_time,
    params.seq_lens_decoder,
    params.seq_lens_encoder,
    params.padding_offsets,
    params.O,
    params.max_seq_len,
    params.chunk_num,
    params.q_num_head,
    params.chunk_size,
    params.vo_head_dim,
    params.token_num,
    params.bsz,
    params.max_draft_token_num
  );
#ifdef DEBUG_MLA
  cudaDeviceSynchronize();
  auto err = cudaGetLastError();
    printf("err = %d, str = %s\n", err, cudaGetErrorString(err));
#endif
  return cudaSuccess;
}

template <uint32_t HEAD_DIM_QK, uint32_t HEAD_DIM_VO, MaskMode MASK_MODE, typename Params>
cudaError_t BatchMLAWithPagedKVCacheDispatched(Params& params, cudaStream_t stream) {
  constexpr bool CAUSAL = MASK_MODE == MaskMode::kCausal;
  if constexpr (HEAD_DIM_QK == 576) {
#ifdef DEBUG_MLA
    printf("\ngoto HEAD_DIM_QK 576\n");
#endif
    // NOTE(Zihao): CTA_KV not tuned for HEAD_DIM == 64, need to optimize later
    DISPATCH_GQA_GROUP_SIZE(params.q_num_head, GROUP_SIZE,
      BatchMLAWithPagedKVCacheKernelTraitsDispatched<
          AttentionKernelTraits</*USE_TMA_LOAD_KV=*/false, HEAD_DIM_QK, HEAD_DIM_VO, GROUP_SIZE,
                                /*CTA_Q_=*/64,
                                /*CTA_KV_=*/64,
                                /*NUM_STAGES_=*/2, typename Params::DTypeQ,
                                typename Params::DTypeKV, typename Params::DTypeO,
                                typename Params::IdType>,
          CAUSAL>(params, stream);)
  } else {
    return cudaErrorNotSupported;
  }
  cudaError_t status = cudaGetLastError();
  return status;
};

}  // namespace flashinfer

#endif  // FLASHINFER_ATTENTION_HOPPER_PREFILL_SM90_CUH_
