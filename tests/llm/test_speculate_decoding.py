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
from __future__ import annotations

import os
import unittest

import paddle

from paddlenlp.transformers import (  # ChatGLMForCausalLM,
    AutoTokenizer,
    LlamaForCausalLM,
)

from .testing_utils import LLMTest, argv_context_guard, load_test_config


class SpeculatePredictorTest(LLMTest, unittest.TestCase):
    config_path: str = "./tests/fixtures/llm/predictor.yaml"
    model_name_or_path: str = "__internal_testing__/tiny-random-llama-hd128"
    model_class = LlamaForCausalLM

    def setUp(self) -> None:
        super().setUp()
        paddle.set_default_dtype("bfloat16")
        self.model_class.from_pretrained(self.model_name_or_path, dtype="bfloat16").save_pretrained(self.output_dir)
        AutoTokenizer.from_pretrained(self.model_name_or_path).save_pretrained(self.output_dir)

    def run_predictor(self, config_params=None):
        if config_params is None:
            config_params = {}

        # to avoid the same parameter
        self.disable_static()
        predict_config = load_test_config(self.config_path, "inference-predict")
        predict_config["output_file"] = os.path.join(self.output_dir, "predict.json")
        predict_config["model_name_or_path"] = self.output_dir
        predict_config.pop("data_file")
        predict_config.pop("decode_strategy")
        predict_config.update(config_params)

        with argv_context_guard(predict_config):
            from predict.predictor import predict

            predict()

        # prefix_tuning dynamic graph do not support to_static
        if not predict_config["inference_model"]:
            return

        # to static
        self.disable_static()
        config = load_test_config(self.config_path, "inference-to-static")
        config["output_path"] = self.inference_output_dir
        config["model_name_or_path"] = self.output_dir
        config.update(config_params)

        with argv_context_guard(config):
            from predict.export_model import main

            main()
        # inference
        self.disable_static()
        config = load_test_config(self.config_path, "inference-infer")
        config["model_name_or_path"] = self.inference_output_dir
        config["output_file"] = os.path.join(self.inference_output_dir, "infer.json")
        config.pop("data_file")
        config.pop("decode_strategy")
        config_params.pop("model_name_or_path", None)
        config.update(config_params)

        with argv_context_guard(config):
            from predict.predictor import predict

            predict()

        self.disable_static()

        predict_result = self._read_result(predict_config["output_file"])
        infer_result = self._read_result(config["output_file"])
        assert len(predict_result) == len(infer_result)

    def test_forward(self):
        self.run_predictor(
            {
                "inference_model": True,
                "src_length": 512,
                "max_length": 48,
                "speculate_method": "inference_with_reference",
                "speculate_max_draft_token_num": 5,
                "speculate_max_ngram_size": 2,
            }
        )


if __name__ == "__main__":
    unittest.main()
