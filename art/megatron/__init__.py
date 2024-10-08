# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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
#

# from .global_vars import get_args
# from .global_vars import get_tokenizer
# from .global_vars import get_t0_tokenizer
# from .global_vars import get_t0_model
# from .global_vars import get_evidence_in_string
# from .global_vars import get_tensorboard_writer
# from .global_vars import get_timers
# from .initialize  import initialize_megatron

import torch
import os
def print_rank_0(message):
    """If distributed is initialized print only on rank 0."""
    if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
            print(message, flush=True)
    else:
        local_rank = int(os.environ["LOCAL_RANK"])
        if local_rank == 0:
            print(message, flush=True)
