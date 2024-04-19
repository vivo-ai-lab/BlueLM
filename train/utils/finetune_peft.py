# Copyright 2023 vivo.
#
# Copyright 2021 The HuggingFace Team. All rights reserved.
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
from typing import List

import torch

from peft import (
    get_peft_model,
    LoraConfig,
    TaskType,
)


def find_all_linear_modules(model: "PreTrainedModel", logger) -> List[str]:
    linear_cls = torch.nn.Linear
    output_layer_names = ["lm_head"]
    module_names = set()
    for name, module in model.named_modules():
        if (
            isinstance(module, linear_cls)
            and not any([output_layer in name for output_layer in output_layer_names])
        ):
            module_names.add(name.split(".")[-1])
    logger.info("Found linear modules: {}".format(",".join(module_names)))
    return list(module_names)


def get_peft_config(model, args, logger):
    lora_target = [item.strip() for item in args.lora_target.split(",")]
    if len(lora_target) == 1 and lora_target[0] == "all":
        target_modules = find_all_linear_modules(model, logger)
    else:
        target_modules = args.lora_target
    return LoraConfig(
            task_type=TaskType.CAUSAL_LM, inference_mode=False,
            r=args.lora_rank,
            lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
            target_modules=target_modules
        )


def get_peft_lora_model(model, args, logger=None):
    peft_config = get_peft_config(model, args, logger)
    return get_peft_model(model, peft_config)
