#include "paddle/extension.h"
#include "remote_cache_kv_ipc.h"
#include "paddle/phi/core/allocator.h"
#include "paddle/phi/core/dense_tensor.h"

using cache_write_compelete_signal_type = RemoteCacheKvIpc::save_cache_kv_compelete_signal_layerwise_meta_data;

std::vector<paddle::Tensor> OpenShmAndGetMetaSignal(const int rank) {
    cache_write_compelete_signal_type kv_signal_metadata;
    const char* fmt_write_cache_completed_signal_str = std::getenv("FLAGS_fmt_write_cache_completed_signal");
    if (fmt_write_cache_completed_signal_str && 
        (std::strcmp(fmt_write_cache_completed_signal_str, "true") == 0 ||
         std::strcmp(fmt_write_cache_completed_signal_str, "1") == 0)) {
        kv_signal_metadata = RemoteCacheKvIpc::open_shm_and_get_compelete_signal_meta_data(rank);
    }

    auto kv_signal_metadata_out = paddle::full({3}, -1, paddle::DataType::INT64, paddle::CPUPlace());
    kv_signal_metadata_out.data<int64_t>()[0] = static_cast<int64_t>(kv_signal_metadata.layer_id);
    kv_signal_metadata_out.data<int64_t>()[1] = reinterpret_cast<int64_t>(kv_signal_metadata.shm_ptr);
    kv_signal_metadata_out.data<int64_t>()[2] = static_cast<int64_t>(kv_signal_metadata.shm_fd);
    return {kv_signal_metadata_out};
}


std::vector<std::vector<int64_t>> OpenShmAndGetMetaSignalShape(const int rank) {
    return {{3}};
}

std::vector<paddle::DataType> OpenShmAndGetMetaSignalDtype(const int rank) {
    return {paddle::DataType::INT64};
}

PD_BUILD_OP(open_shm_and_get_meta_signal)
    .Inputs({})
    .Outputs({"kv_signal_metadata"})
    .Attrs({"rank: int"})
    .SetKernelFn(PD_KERNEL(OpenShmAndGetMetaSignal))
    .SetInferShapeFn(PD_INFER_SHAPE(OpenShmAndGetMetaSignalShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(OpenShmAndGetMetaSignalDtype));