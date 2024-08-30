# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from paddlenlp_ops import tune_cublaslt_gemm
import paddle

M_tensor = paddle.to_tensor([32768])

# qwen2-1.5b
k3 = [1536, 1536, 1536, 8960]
n3 = [2048, 1536, 17920, 1536]

K_tensor = paddle.to_tensor(k3)
N_tensor = paddle.to_tensor(n3)

Dtype = "int8"
Path = "./cublaslt_gemm_search_3.csv"

tune_cublaslt_gemm(M_tensor, K_tensor, N_tensor, Dtype, True, False, Path)
