/*
 * Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri
 * Dao. Licensed under the BSD 3-Clause.
 *
 * Modified by the FlashInfer team.
 */
#ifndef FLASHINFER_ATTENTION_HOPPER_MAINLOOP_MMA_CUH_
#define FLASHINFER_ATTENTION_HOPPER_MAINLOOP_MMA_CUH_

#include <cutlass/array.h>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/numeric_types.h>
#include "named_barrier.cuh"

namespace flashinfer {

template <typename Ktraits, bool CAUSAL, typename Params, typename MainloopPipeline,
          typename PipelineState, typename SharedStorage, typename FrgTensorO,
          typename AttentionUpdater>
CUTLASS_DEVICE void mma_f16(const Params& mainloop_params,
                            MainloopPipeline pipeline_q,
                            PipelineState& smem_pipe_read_q,
                            MainloopPipeline pipeline_kv,
                            PipelineState& smem_pipe_read_kv,
                            FrgTensorO& tOrO, 
                            AttentionUpdater& attention_updater,
                            const int thread_idx, 
                            const int bid,
                            const int kv_len,
                            const int qo_len,
                            const int tile_idx,
                            SharedStorage& shared_storage) {
  using DTypeQ = typename Ktraits::DTypeQ;
  using DTypeKV = typename Ktraits::DTypeKV;
  using DTypeMD = typename Ktraits::DTypeO; // !!! bf16
  using DTypeQKAccum = typename Ktraits::DTypeQKAccum;
  using IdType = typename Ktraits::IdType;
  using TileShape_QKD = typename Ktraits::TileShape_QKD;
  static constexpr int NUM_MMA_THREADS = Ktraits::NUM_MMA_THREADS;
  using SmemLayoutQ = typename Ktraits::SmemLayoutQ;
  using SmemLayoutK = typename Ktraits::SmemLayoutK;
  using SmemLayoutV = typename Ktraits::SmemLayoutV;
  using SmemLayoutP = typename Ktraits::SmemLayoutP;
  using SmemLayoutRow = typename Ktraits::SmemLayoutRow;
  using SmemCopyAtom = typename Ktraits::SmemCopyAtom;
  using SmemLayoutVt = typename Ktraits::SmemLayoutVt;
  using SmemLayoutVtOneStage = typename Ktraits::SmemLayoutVtOneStage;
  // using RealSmemLayoutV = decltype(make_layout(get<0>(SmemLayoutVtTest{}), get<1>(SmemLayoutVtTest{}), get<2>(SmemLayoutVt{})));
  static_assert(is_rmem<FrgTensorO>::value, "O tensor must be rmem resident.");

  static constexpr int CTA_Q = get<0>(TileShape_QKD{});
  static constexpr int CTA_KV = get<1>(TileShape_QKD{});

  Tensor sQ = make_tensor(make_smem_ptr(shared_storage.smem_q.data()), SmemLayoutQ{});
  Tensor sK = make_tensor(make_smem_ptr(shared_storage.smem_kv.data()), SmemLayoutK{});
  Tensor sVt_s1 = make_tensor(make_smem_ptr(shared_storage.smem_kv.data()), SmemLayoutVtOneStage{});
  Tensor sVt_s2 = make_tensor(make_smem_ptr(shared_storage.smem_kv.data() + Ktraits::NUM_PER_STAGE), SmemLayoutVtOneStage{});
  Tensor sP = make_tensor(make_smem_ptr(shared_storage.smem_p.data()), SmemLayoutP{});
  Tensor s_scale = make_tensor(make_smem_ptr(shared_storage.smem_scale.data()), SmemLayoutRow{});
  Tensor mM = make_tensor(make_gmem_ptr(mainloop_params.m_ptr), mainloop_params.layout_MD)(tile_idx, _); // (bsz * draft_token_num * num_head)
  Tensor mD = make_tensor(make_gmem_ptr(mainloop_params.d_ptr), mainloop_params.layout_MD)(tile_idx, _);
#ifdef DEBUG_MLA
  if (thread(128)) {
    printf("\nmM: \n");
    print(mM);
    printf("\nmD: \n");
    print(mD);
    printf("\nrow_MAX: \n");
    print(attention_updater.row_max);
    printf("\nrow_SUM: \n");
    print(attention_updater.row_sum);
  }
#endif

  typename Ktraits::TiledMmaQK tiled_mma_qk;
  auto threadMmaQK = tiled_mma_qk.get_thread_slice(thread_idx);
  auto smem_tiled_copy_P = make_tiled_copy_C(SmemCopyAtom{}, tiled_mma_qk);
  auto smem_thr_copy_P = smem_tiled_copy_P.get_thread_slice(thread_idx);
  Tensor tPsP = smem_thr_copy_P.partition_D(sP);
  Tensor tScalesScale = s_scale(_, thread_idx);

  typename Ktraits::TiledMmaPV tiled_mma_pv;
  auto threadMmaPV = tiled_mma_pv.get_thread_slice(thread_idx);
  Tensor tOrV1 = threadMmaPV.partition_fragment_B(sVt_s1);
  Tensor tOrV2 = threadMmaPV.partition_fragment_B(sVt_s2);
  Tensor tOrP_CS2 = threadMmaPV.partition_fragment_A(sP);

  // int num_kv_tiles = cute::ceil_div(kv_len, CTA_KV);
  const int start_len = tile_idx * mainloop_params.chunk_size;
  const int start_tile_idx = start_len / CTA_KV;
  const int end_tile_idx = min(start_len + mainloop_params.chunk_size, kv_len) / CTA_KV;
  int kv_tile_idx = end_tile_idx;

  auto consumer_wait = [](auto& pipeline, auto& smem_pipe_read) {
    auto barrier_token = pipeline.consumer_try_wait(smem_pipe_read);
    pipeline.consumer_wait(smem_pipe_read, barrier_token);
  };

  int count = 0;
  int warp_group_idx = cutlass::canonical_warp_group_idx();
  // kv is ready
  cutlass::arch::NamedBarrier::arrive(Ktraits::NUM_THREADS,
                                      /*id=*/static_cast<int>(NamedBarriers::kOdone));
  if (warp_group_idx == 1) {
    // consumer 0, compute qk
    Tensor tSrQ = threadMmaQK.partition_fragment_A(sQ);
    Tensor tSrK = threadMmaQK.partition_fragment_B(sK);
    // DO QK gemm
#ifdef DEBUG_MLA
    if (thread(128)) {
      printf("\ntiled_mma_qk:\n");
      print(tiled_mma_qk);
      printf("\ntiled_mma_pv:\n");
      print(tiled_mma_pv);
      printf("\nsQ:\n");
      print(sQ);
      printf("\nsK:\n");
      print(sK);
      printf("\nsVt one stage:\n");
      print(sVt_s1);
      printf("\nsP:\n");
      print(sP);
      printf("\ns_scale:\n");
      print(s_scale);
      printf("\ntSrQ:\n");
      print(tSrQ);
      printf("\ntSrK:\n");
      print(tSrK);
      printf("\ntOrV:\n");
      print(tOrV1);
      printf("\ntOrO:\n");
      print(tOrO);
    }
#endif
    constexpr int n_masking_steps = !CAUSAL ? 1 : cute::ceil_div(CTA_Q, CTA_KV) + 1;
    auto col_limit_right = [&](int qo_idx) { return qo_idx + 1 + kv_len - qo_len; };
    bool is_first_step = true;
    // wait q
    consumer_wait(pipeline_q, smem_pipe_read_q);
#pragma unroll 1
    for (int masking_step = n_masking_steps; kv_tile_idx >= start_tile_idx; --masking_step, --kv_tile_idx) {
#ifdef DEBUG_MLA
      if (thread(128)) {
        printf("wg1 111\n");
      }
#endif
      Tensor tSrS = partition_fragment_C(tiled_mma_qk, select<0, 1>(TileShape_QKD{}));
      // wait kv
      consumer_wait(pipeline_kv, smem_pipe_read_kv);
      // gemm qk
      gemm</*init=*/true, /*wg_wait=*/0>(tiled_mma_qk, tSrQ, tSrK(_, _, _, smem_pipe_read_kv.index()),
                                         tSrS);
      // mask
      if (masking_step > 0) {
        Tensor cS = cute::make_identity_tensor(select<0, 1>(TileShape_QKD{}));
        Tensor tScS = threadMmaQK.partition_C(cS);
#pragma unroll
        for (int i = 0; i < size(tSrS); ++i) {
          int qo_idx = get<0>(tScS(i)) / Ktraits::GROUP_SIZE;
          int kv_idx = get<1>(tScS(i)) + kv_tile_idx * CTA_KV;
          if constexpr (!CAUSAL) {  // Just masking based on col
            if (kv_idx >= kv_len) {
              tSrS(i) = AttentionUpdater::fill_value;
            }
          } else {
            if (kv_idx >= std::min(kv_len, col_limit_right(qo_idx))) {
              tSrS(i) = AttentionUpdater::fill_value;
            }
          }
        }
      }
      // update s (exp(s - m))
      Tensor scale_o = is_first_step ? attention_updater.update</*init=*/true>(tSrS) : attention_updater.update</*init=*/false>(tSrS);
      is_first_step = false;

      Tensor convert_tSrS = convert_type<DTypeKV>(tSrS);
      Tensor tPrP = smem_thr_copy_P.retile_S(convert_tSrS);
#ifdef DEBUG_MLA
      if (thread(128)) {
        printf("\ntSrS: \n");
        print(tSrS);
        printf("\nconvert_tSrS: \n");
        print(convert_tSrS);
        printf("\ntPrP: \n");
        print(tPrP);
        printf("\ntPsP: \n");
        print(tPsP);
        printf("\nscale_o: \n");
        print(scale_o);
      }
#endif
      // gather qk gemm res
      cute::copy(smem_tiled_copy_P, tPrP, tPsP);
      cute::copy(scale_o, tScalesScale);
#ifdef DEBUG_MLA
      if (thread(128)) {
        printf("wg1 122, NUM_MMA_THREADS: %d\n", (int)Ktraits::NUM_MMA_THREADS);
      }
#endif
      cutlass::arch::NamedBarrier::arrive(Ktraits::NUM_MMA_THREADS, static_cast<int>(NamedBarriers::kWarpSchedulerWG1));
#ifdef DEBUG_MLA
      if (thread(128)) {
        printf("wg1 222\n"); 
      }
#endif
      attention_updater.rescale_o(tOrO, scale_o);
      Tensor tOrP = make_tensor(convert_tSrS.data(),
                                convert_layout_acc_Aregs<typename Ktraits::TiledMmaPV>(tSrS.layout()));
#ifdef DEBUG_MLA
      if (thread(128)) {
        printf("\ntOrP: \n");
        print(tOrP);
        printf("\ntOrP_CS2: \n");
        print(tOrP_CS2);
        printf("\ntPrP: \n");
        print(tPrP);
        printf("\ntPsP: \n");
        print(tPsP);
        printf("\nscale_o: \n");
        print(scale_o);
      }
#endif
      // pv gemm
      if (count % 2 == 0) {
        gemm</*init=*/false, /*wg_wait=*/0>(tiled_mma_pv, tOrP,
                                            tOrV1(_, _, _, _0{}), tOrO);
      } else {
        gemm</*init=*/false, /*wg_wait=*/0>(tiled_mma_pv, tOrP,
                                            tOrV2(_, _, _, _0{}), tOrO);
      }
      ++count;
      cutlass::arch::NamedBarrier::sync(Ktraits::NUM_MMA_THREADS, static_cast<int>(NamedBarriers::kWarpSchedulerWG2));
#ifdef DEBUG_MLA
      if (thread(128)) {
        printf("wg1 333\n");
      }
#endif
      pipeline_kv.consumer_release(smem_pipe_read_kv);
      ++smem_pipe_read_kv;
    } 
    // WG1 write m,d back to gmem
    if (thread_idx % 4 == 0) { // 16 rows per warp, eg. t0->row0 row8，t4->row1 row9
      const int warp_idx = thread_idx / 32;
#pragma unroll
      for (int w_i = 0; w_i < 2; ++w_i) {
        const int token_group_idx = warp_idx * 16 + thread_idx / 4 + 8 * w_i;
        const int token_idx = token_group_idx / Ktraits::GROUP_SIZE;
#ifdef DEBUG_MLA
        if (thread(128)) {
          printf("\ntoken_group_idx: %d\n", token_group_idx);
          print(token_group_idx);
          printf("\ntoken_idx: %d\n", token_idx);
          print(mD);
          printf("\nrow_MAX: \n");
          print(attention_updater.row_max);
          printf("\nrow_SUM: \n");
          print(attention_updater.row_sum);
        }
#endif
        if (token_idx < qo_len) {
          const int head_idx = token_group_idx % Ktraits::GROUP_SIZE;
          const int bid_offset = mainloop_params.max_draft_token_num * Ktraits::GROUP_SIZE;
          const int write_idx = bid * bid_offset + token_idx * Ktraits::GROUP_SIZE + head_idx;
          mM(write_idx) = static_cast<DTypeMD>(attention_updater.row_max(w_i));
          mD(write_idx) = static_cast<DTypeMD>(attention_updater.row_sum(w_i));
        }
      }
    }


  } else if (warp_group_idx == 2) {
    // consumer 1, compute pv
    for (; kv_tile_idx >= start_tile_idx; --kv_tile_idx) {
      Tensor scale_o = make_tensor<DTypeQKAccum>(Shape<_2>{});
      cutlass::arch::NamedBarrier::sync(Ktraits::NUM_MMA_THREADS, static_cast<int>(NamedBarriers::kWarpSchedulerWG1));
#ifdef DEBUG_MLA
      if (thread(256)) {
        printf("wg2 111\n");
      }
#endif
      // A: tPsP
      cute::copy(tScalesScale, scale_o);
      // rescale
      attention_updater.rescale_o(tOrO, scale_o);
      if (count % 2 == 0) {
        gemm</*init=*/false, /*wg_wait=*/0>(tiled_mma_pv, tOrP_CS2,
                                            tOrV1(_, _, _, _0{}), tOrO);
      } else {
        gemm</*init=*/false, /*wg_wait=*/0>(tiled_mma_pv, tOrP_CS2,
                                            tOrV2(_, _, _, _0{}), tOrO);
      }
      ++count;
      cutlass::arch::NamedBarrier::sync(Ktraits::NUM_MMA_THREADS, static_cast<int>(NamedBarriers::kWarpSchedulerWG2));
#ifdef DEBUG_MLA
      if (thread(256)) {
        printf("wg2 222\n");
      }
#endif
      pipeline_kv.consumer_release(smem_pipe_read_kv);
      ++smem_pipe_read_kv;
    }
  }
  pipeline_q.consumer_release(smem_pipe_read_q);
  ++smem_pipe_read_q;
  return;
}

}  // namespace flashinfer

#endif  // FLASHINFER_ATTENTION_HOPPER_MAINLOOP_MMA_CUH_
