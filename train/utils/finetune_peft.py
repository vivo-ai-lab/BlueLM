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


from peft import (
    get_peft_model,
    LoraConfig,
    TaskType,
)

def get_peft_config(args):
    return LoraConfig(
            task_type=TaskType.CAUSAL_LM, inference_mode=False,
            r=args.lora_rank,
            lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
            target_modules=['q_proj', 'v_proj']
        )

def get_peft_lora_model(model, args):
    peft_config = get_peft_config(args)
    return get_peft_model(model, peft_config)