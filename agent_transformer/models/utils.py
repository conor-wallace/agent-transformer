from typing import Tuple

import copy
import torch
import torch.nn as nn


def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


def stack_inputs(embeddings: Tuple[torch.Tensor]) -> torch.Tensor:
    batch_size = embeddings[0].shape[0]
    seq_length = embeddings[0].shape[1]
    hidden_dim = embeddings[0].shape[2]
    num_modalities = len(embeddings)

    # this makes the sequence look like (R_1, s_1, a_1, R_2, s_2, a_2, ...)
    # which works nice in an autoregressive sense since states predict actions
    stacked_inputs = torch.stack(embeddings, dim=1).permute(0, 2, 1, 3).reshape(batch_size, num_modalities * seq_length, hidden_dim)

    return stacked_inputs


def stack_attention_mask(attention_mask, num_modalities, batch_size, seq_length):
    # to make the attention mask fit the stacked inputs, have to stack it as well
    stacked_attention_mask = torch.stack(
        (attention_mask,) * num_modalities, dim=1
    ).permute(0, 2, 1).reshape(batch_size, num_modalities * seq_length)

    return stacked_attention_mask