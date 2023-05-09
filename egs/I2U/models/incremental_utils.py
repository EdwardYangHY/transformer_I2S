import torch
import torch.nn.functional as F
# from torch.nn.functional import (
#     _mha_shape_check, _in_projection_packed, _in_projection, pad, softmax, dropout, linear
#     )
# from torch.nn.modules.activation import MultiheadAttention
import uuid
from typing import Dict, Optional

from torch import Tensor
# from torch.overrides import (
#     has_torch_function, has_torch_function_unary, has_torch_function_variadic,
#     handle_torch_function)

"""
    get_incremental_state
    set_incremental_state
    reorder_incremental_state
    based on layer_ids
"""

"""
    Data structure for Incremental state is:
    {
        layer_1_id:{
            "prev_key": Tensor
            "prev_value": Tensor
            "prev_mask": Tensor
        },
        layer_2_id:{},
        ...
        layer_n_id:{},
    }
"""


class IncrementalState(object):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.init_incremental_state()

    def init_incremental_state(self, incremental_state, layer_id):
        if layer_id not in incremental_state.keys():
            incremental_state[layer_id] = {}
        return
    
    def get_incremental_state(self, incremental_state, layer_id):
        return incremental_state[layer_id]
    
    def set_incremental_state(self, incremental_state, saved_state, layer_id = None):
        for k, v in saved_state.items():
            if layer_id is not None:
                incremental_state[layer_id][k] = v
            else:
                incremental_state[k] = v
    
    def reoder_incremental_state(
            self,
            incremental_state,
            new_order
    ):
        """
            Few things to do:
            Given a new order of sequences, it should reorder all cached keys and values.
            Also, it should cut sizes of keys and values when some sentences are complete.

            It is often operated outside MHA. Probably this function in this class will not be used.
        """
        pass

# @with_incremental_state
# as decorator, so the decorated class would also obtain those functions.
def with_incremental_state(cls):
    cls.__bases__ = (IncrementalState,) + tuple(
        b for b in cls.__bases__ if b != IncrementalState
    )
    return cls