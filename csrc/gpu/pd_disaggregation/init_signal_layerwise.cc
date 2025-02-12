#include "paddle/extension.h"
#include "remote_cache_kv_ipc.h"
#include "paddle/phi/core/allocator.h"
#include "paddle/phi/core/dense_tensor.h"

using cache_write_compelete_signal_type = RemoteCacheKvIpc::save_cache_kv_compelete_signal_layerwise_meta_data;

std::vector<paddle::Tensor> InitSignalLayerwise(const paddle::Tensor& kv_signal_metadata, const int layer_id) {
    auto kv_signal_metadata_out = kv_signal_metadata.copy_to(paddle::CPUPlace(), false);
    kv_signal_metadata_out.data<int64_t>()[0] = static_cast<int64_t>(layer_id);
    return {kv_signal_metadata_out};
}

std::vector<std::vector<int64_t>> InitSignalLayerwiseShape(
    const std::vector<int64_t>& kv_signal_metadata_shape,
    const int layer_id) {
    return {kv_signal_metadata_shape};
}

std::vector<paddle::DataType> InitSignalLayerwiseDtype(
    const paddle::DataType& kv_signal_metadata_dtype,
    const int layer_id) {
    return {paddle::DataType::INT64};
}

PD_BUILD_OP(init_signal_layerwise)
    .Inputs({"kv_signal_metadata"})
    .Outputs({"kv_signal_metadata_out"})
    .Attrs({"layer_id: int"})
    .SetKernelFn(PD_KERNEL(InitSignalLayerwise))
    .SetInferShapeFn(PD_INFER_SHAPE(InitSignalLayerwiseShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(InitSignalLayerwiseDtype));