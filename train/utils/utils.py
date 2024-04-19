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

import os
import logging
import sys
import torch
import argparse


def get_optimizer_grouped_parameters(
        model,
        weight_decay,
        lora_lr=5e-4,
        no_decay_name_list=["bias", "LayerNorm.weight"],
        lora_name_list=["lora_right_weight", "lora_left_weight"],
):
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters()
                if (not any(nd in n for nd in no_decay_name_list)
                    and p.requires_grad and not any(nd in n
                                                    for nd in lora_name_list))
            ],
            "weight_decay":
                weight_decay,
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if (not any(nd in n for nd in no_decay_name_list)
                    and p.requires_grad and any(nd in n
                                                for nd in lora_name_list))
            ],
            "weight_decay":
                weight_decay,
            "lr":
                lora_lr
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if (any(nd in n
                        for nd in no_decay_name_list) and p.requires_grad)
            ],
            "weight_decay":
                0.0,
        },
    ]
    if not optimizer_grouped_parameters[1]["params"]:
        optimizer_grouped_parameters.pop(1)
    return optimizer_grouped_parameters


def to_device(batch):
    if torch.is_tensor(batch):
        return batch.cuda(non_blocking=True)
    if isinstance(batch, list):
        return [to_device(item) for item in batch]
    if isinstance(batch, tuple):
        return (to_device(item) for item in batch)
    if isinstance(batch, dict):
        result = {}
        for key in batch:
            result[key] = to_device(batch[key])
        return result
    return batch


def to_half(batch):
    if torch.is_tensor(batch):
        if torch.is_floating_point(batch):
            return batch.half()
        else:
            return batch
    if isinstance(batch, list):
        return [to_half(item) for item in batch]
    if isinstance(batch, tuple):
        return (to_half(item) for item in batch)
    if isinstance(batch, dict):
        result = {}
        for key in batch:
            result[key] = to_half(batch[key])
        return result
    return batch


def to_bf16(batch):
    if torch.is_tensor(batch):
        if torch.is_floating_point(batch):
            return batch.bfloat16()
        else:
            return batch
    if isinstance(batch, list):
        return [to_bf16(item) for item in batch]
    if isinstance(batch, tuple):
        return (to_bf16(item) for item in batch)
    if isinstance(batch, dict):
        result = {}
        for key in batch:
            result[key] = to_bf16(batch[key])
        return result
    return batch


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.local_rank = int(os.environ['LOCAL_RANK'])
        args.world_size = int(os.environ['WORLD_SIZE'])
    else:
        args.rank = 0
        args.local_rank = 0
        args.world_size = 1
        setup_for_distributed(is_master=True)  # hack
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.local_rank)
    if args.deepspeed:
        import deepspeed
        deepspeed.init_distributed()
    else:
        torch.distributed.init_process_group(backend='nccl', init_method='env://',
                                             world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)
    return

class AmpScaler:
    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = torch.cuda.amp.GradScaler()

    def __call__(self, loss, optimizer, lr_scheduler, create_graph=False, update_grad=True):
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if update_grad:
            self._scaler.unscale_(optimizer)
            self._scaler.step(optimizer)
            self._scaler.update()
            lr_scheduler.step()

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs',
                        type=int,
                        default=1,
                        help="The count of training epoch")
    parser.add_argument('--deepspeed',
                        action='store_true',
                        help="Whether use deepspeed to train")
    parser.add_argument('--deepspeed_config',
                        type=str,
                        default=None,
                        help="The path of deepspeed config file")
    parser.add_argument('--gradient_checkpointing',
                        action='store_true',
                        help="Whether use gradient checkpointing")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="The step of gradient accumulation")
    parser.add_argument('--do_validate',
                        action='store_true',
                        help="Whether validate during train process")
    parser.add_argument('--train_file',
                        type=str,
                        default=None,
                        help="The path of train file")
    parser.add_argument('--cache_dir',
                        type=str,
                        default=None,
                        help="Where to store the pretrained models downloaded from huggingface.co")
    parser.add_argument('--max_source_length',
                        type=int,
                        default=1024,
                        help="The maximum total input sequence length after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded.")
    parser.add_argument('--max_target_length',
                        type=int,
                        default=1024,
                        help="The maximum total sequence length for target text after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded.")
    parser.add_argument('--prompt_column',
                        type=str,
                        default="inputs",
                        help="The name of prompt column")
    parser.add_argument('--response_column',
                        type=str,
                        default="targets",
                        help="The name of response column")
    parser.add_argument('--max_steps',
                        type=int,
                        default=0,
                        help="The max steps for training")
    parser.add_argument('--save_steps',
                        type=int,
                        default=1000,
                        help="The interval of model saving")
    parser.add_argument('--model_name_or_path',
                        type=str,
                        default=None,
                        help="The path of model")
    parser.add_argument('--output_dir',
                        type=str,
                        default=None,
                        help="The output model saving directory")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="The initial seed")
    parser.add_argument('--seq_len',
                        type=int,
                        default=2048,
                        help="The length of input sequence")
    parser.add_argument('--finetune',
                        action='store_true',
                        help="Whether in pretrain of finetune stage")
    parser.add_argument('--batch_size_per_device',
                        type=int,
                        default=1,
                        help="The data batch size per device for training")
    parser.add_argument('--local_rank',
                        type=int,
                        default=0,
                        help="The rank of local device")
    parser.add_argument('--learning_rate',
                        type=float,
                        default=5e-5,
                        help="The learning rate of training")
    parser.add_argument('--weight_decay',
                        type=float,
                        default=0.0,
                        help="Weight decay for AdamW if we apply some.")
    parser.add_argument('--tensorboard_dir',
                        type=str,
                        default=None,
                        help="The tensorboard info output directory.")
    parser.add_argument('--lora_rank',
                        type=int,
                        default=None,
                        help='LoRA rank')
    parser.add_argument('--lora_alpha',
                        type=int,
                        default=16,
                        help='LoRA alpha')
    parser.add_argument('--lora_dropout',
                        type=float,
                        default=0.05,
                        help='LoRA dropout'
                        )
    parser.add_argument('--lora_target',
                        type=str,
                        default="all",
                        help='LoRA target'
                        )

    args = parser.parse_args()
    return args


def create_logger(name=None, level=logging.INFO, rank=0):
    """create a logger

    Args:
        name (str): name of the logger
        level: level of logger

    Raises:
        ValueError is name is None
    """

    if name is None:
        raise ValueError("name for logger cannot be None")

    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] "
                                  "[%(filename)s:%(lineno)d:%(funcName)s] %(message)s")

    logger_ = logging.getLogger(name)
    if rank <= 0:
        logger_.setLevel(level)
        logger_.propagate = False
        ch = logging.StreamHandler(stream=sys.stdout)
        ch.setLevel(level)
        ch.setFormatter(formatter)
    else:
        ch = logging.NullHandler
    logger_.addHandler(ch)
    return logger_
