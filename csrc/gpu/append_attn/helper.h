// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#pragma once
#include "paddle/extension.h"
#include "paddle/phi/core/allocator.h"


template <paddle::DataType D>
class PDTraits;

template <>
class PDTraits<paddle::DataType::FLOAT32> {
public:
  typedef float DataType;
  typedef float data_t;
};

template <>
class PDTraits<paddle::DataType::FLOAT16> {
public:
  typedef half DataType;
  typedef paddle::float16 data_t;
};

template <>
class PDTraits<paddle::DataType::BFLOAT16> {
 public:
  typedef __nv_bfloat16 DataType;
  typedef paddle::bfloat16 data_t;
};

#define CUDA_CHECK(call)                           \
  do {                                             \
    const cudaError_t error_code = call;           \
    if (error_code != cudaSuccess) {               \
      std::printf("at %s:%d - %s.\n",              \
                  __FILE__,                        \
                  __LINE__,                        \
                  cudaGetErrorString(error_code)); \
      exit(1);                                     \
    }                                              \
  } while (0)