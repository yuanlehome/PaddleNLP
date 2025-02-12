// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
// 
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// 
//     http://www.apache.org/licenses/LICENSE-2.0
// 
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "helper.h"
#include<stdlib.h>
#include<string.h>
#include<sys/types.h>
#include<sys/stat.h>
#include<unistd.h>
#include<fcntl.h>
#include<sys/mman.h>
#include<stdio.h>
#include "cuda_multiprocess_helper.h"
#include "paddle/phi/core/tensor_meta.h"


void ShareExternalData(paddle::Tensor& input,
                       const std::string shm_name) {     
  volatile shmStruct *shm = NULL;
  sharedMemoryInfo info;
  if (sharedMemoryOpen(shm_name.c_str(), sizeof(shmStruct), &info) != 0) {
    printf("Failed to create shared memory slab\n");
    exit(EXIT_FAILURE);
  }
  shm = (volatile shmStruct *)info.addr;
  void *ptr = nullptr;
  checkCudaErrors(
      cudaIpcOpenMemHandle(&ptr,
                           *(cudaIpcMemHandle_t *)&shm->memHandle,  // NOLINT
                           cudaIpcMemLazyEnablePeerAccess));

  auto shape = input.shape();
  // NOTE(Zhenyu Li): Unable to enter the correct branch when using enum
  // types(why???)) 22: bfloat16, 10: float32, 15: float16, 3: int8, 2: uint8
  if (input.type() == paddle::DataType::BFLOAT16) {
    size_t size = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>()) * sizeof(phi::dtype::bfloat16);
    phi::DenseTensorMeta meta(phi::DataType::BFLOAT16, input.dims(), phi::DataLayout::NCHW);
    std::shared_ptr<phi::DenseTensor> dtensor = std::make_shared<phi::DenseTensor>(std::make_shared<phi::Allocation>(static_cast<phi::dtype::bfloat16*>(ptr), size, input.place()), meta);
    input.set_impl(std::move(dtensor));
  } else if (input.type() == paddle::DataType::FLOAT32) {
    size_t size = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>()) * sizeof(float);
    phi::DenseTensorMeta meta(phi::DataType::FLOAT32, input.dims(), phi::DataLayout::NCHW);
    std::shared_ptr<phi::DenseTensor> dtensor = std::make_shared<phi::DenseTensor>(std::make_shared<phi::Allocation>(static_cast<float*>(ptr), size, input.place()), meta);
    input.set_impl(std::move(dtensor));
  } else if (input.type() == paddle::DataType::FLOAT16) {
    size_t size = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>()) * sizeof(phi::dtype::float16);
    phi::DenseTensorMeta meta(phi::DataType::FLOAT16, input.dims(), phi::DataLayout::NCHW);
    std::shared_ptr<phi::DenseTensor> dtensor = std::make_shared<phi::DenseTensor>(std::make_shared<phi::Allocation>(static_cast<phi::dtype::float16*>(ptr), size, input.place()), meta);
    input.set_impl(std::move(dtensor));
  } else if (input.type() == paddle::DataType::INT8) {
    size_t size = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>()) * sizeof(int8_t);
    phi::DenseTensorMeta meta(phi::DataType::INT8, input.dims(), phi::DataLayout::NCHW);
    std::shared_ptr<phi::DenseTensor> dtensor = std::make_shared<phi::DenseTensor>(std::make_shared<phi::Allocation>(static_cast<int8_t*>(ptr), size, input.place()), meta);
    input.set_impl(std::move(dtensor));
  } else if (input.type() == paddle::DataType::UINT8) {
    size_t size = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>()) * sizeof(uint8_t);
    phi::DenseTensorMeta meta(phi::DataType::UINT8, input.dims(), phi::DataLayout::NCHW);
    std::shared_ptr<phi::DenseTensor> dtensor = std::make_shared<phi::DenseTensor>(std::make_shared<phi::Allocation>(static_cast<uint8_t*>(ptr), size, input.place()), meta);
    input.set_impl(std::move(dtensor));
  } else {
    PADDLE_THROW(phi::errors::Unimplemented(
        "Unsupported data type. Now share_external_data_by_ptr only supports "
        "UINT8, INT8, FLOAT32, BFLOAT16 and FLOAT16"));
  }
  // checkCudaErrors(cudaIpcCloseMemHandle(ptr));
  sharedMemoryClose(&info);
}

PD_BUILD_OP(share_external_data)
    .Inputs({"input"})
    .Outputs({"output"})
    .Attrs({"shm_name: std::string"})
    .SetInplaceMap({{"input", "output"}})
    .SetKernelFn(PD_KERNEL(ShareExternalData));