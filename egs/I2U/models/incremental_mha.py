import sys
import math

import torch
from torch import nn
import torch.nn.functional as F
from typing import Optional, Any, Union, Callable, Tuple
from torch import Tensor
import warnings

from torch.nn.modules.activation import MultiheadAttention
from torch.overrides import (
    has_torch_function, has_torch_function_unary, has_torch_function_variadic,
    handle_torch_function)
from torch.nn.functional import (
    _mha_shape_check, _in_projection_packed, _in_projection, 
    pad, softmax, dropout, linear
    )
from incremental_utils import with_incremental_state
"""
    In this file, we override the Multi-head attention for incremental decoding.
    The tricky part is:
        We should maintain the original weights and bias (params).
        We have to go deep into functionals multihead attention.
        This should be a special mode in inferencing, so it should contain Beam Search.
    
    The key idea of incremental decoding is, when decoding, the calculation of previous 
    information is redundant:
        Each time-step of decoding, we only have one new token added to the previous sequence
        Previous Key and Value can be reused, the calculation of multihead can also be reused.
        So we should store the calculated Key and Value, and process one token each time with
        "incremental state", just like RNN.

    We can do this by simply observe corresponding 

    w_q, w_k, w_v = w.chunk(3)
    b_q, b_k, b_v = b.chunk(3)
"""

"""
    1. Inprojection
        _in_projection_packed can be cached:
        Q, K, V before input projection is: [T, B, E]
        While in incremental, can be [1, B, E]
        After in_proj, Q, K, V remains [1, B, E], with prev_k = [T-1, B, E] and prev_v [T-1, B, E]
        now_k = torch.cat([prev_k, K]) with shape [T, B, E]
        now_v = torch.cat([prev_v, V]) with shape [T, B, E]

        attn_mask: shaped as [tgt_len, src_len] where tgt_len = Q.size(0), src_len = K.size(0)

    2.Reshape for Multihead
        After input projection, the Q, K, V is reshaped to [B * num_heads, T, head_dim], 
        among which head_dim * num_heads = d_model (1024)
        k = k.contiguous().view(k.shape[0], bsz * num_heads, head_dim).transpose(0, 1)

    3.(Deep Breath) calculate attention
        1) q_scaled @ k.transposed(-2, -1)
            attn_output_weights = torch.baddbmm(attn_mask, q_scaled, k.transpose(-2, -1))
            with expect shape [B * num_heads, T, head_dim]
            Now for current:
            attn_output_weights_current = torch.baddbmm(
                                            attn_mask[:,-1:], 
                                            q_scaled[:,-1:], 
                                            k.transpose(-2, -1))
            with expect shape [B * num_heads, 1, head_dim]

            attn_output_weights = softmax()

        2)  attn_output_weight @ v.transposed()
            attn_output = torch.bmm(attn_output_weights, v)
            with expect shape [B * num_heads, T, head_dim] or [B * num_heads, 1, head_dim], depends
            on the given input.
    
    4. Output Projection
        1) Reshape:
            attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len * bsz, embed_dim)
            to [T*B, E]
        2) Linear:
        3) reshape to [T, B, E]
        It seems that, 2), 3) is interchangable.a
"""

@with_incremental_state
class MultiheadAttention_incremental(MultiheadAttention):
    def __init__(
        self, 
        embed_dim, 
        num_heads, 
        # increnmental_state,
        dropout=0, 
        bias=True, 
        add_bias_kv=False, 
        add_zero_attn=False, 
        kdim=None, 
        vdim=None, 
        batch_first=False, 
        device=None, 
        dtype=None
        ) -> None:
        super().__init__(
            embed_dim, 
            num_heads, 
            dropout, 
            bias, 
            add_bias_kv, 
            add_zero_attn, 
            kdim, 
            vdim, 
            batch_first, 
            device, 
            dtype
        )
    def forward(
            self, 
            query: Tensor, 
            key: Tensor, 
            value: Tensor, 
            incremental_state: None,
            key_padding_mask: Optional[Tensor] = None,
            need_weights: bool = True, 
            attn_mask: Optional[Tensor] = None,
            average_attn_weights: bool = True
        ) -> Tuple[Tensor, Optional[Tensor]]:
        # Increamental State is for Each Layer?
        # here the incremental state if only for the present layer
        # with keys: "prev_key", "prev_value"
        
        is_batched = query.dim() == 3
        if key_padding_mask is not None:
            _kpm_dtype = key_padding_mask.dtype
            if _kpm_dtype != torch.bool and not torch.is_floating_point(key_padding_mask):
                raise AssertionError(
                    "only bool and floating types of key_padding_mask are supported")
        why_not_fast_path = ''
        if not is_batched:
            why_not_fast_path = f"input not batched; expected query.dim() of 3 but got {query.dim()}"
        elif query is not key or key is not value:
            # When lifting this restriction, don't forget to either
            # enforce that the dtypes all match or test cases where
            # they don't!
            why_not_fast_path = "non-self attention was used (query, key, and value are not the same Tensor)"
        elif self.in_proj_bias is not None and query.dtype != self.in_proj_bias.dtype:
            why_not_fast_path = f"dtypes of query ({query.dtype}) and self.in_proj_bias ({self.in_proj_bias.dtype}) don't match"
        elif self.in_proj_weight is not None and query.dtype != self.in_proj_weight.dtype:
            # this case will fail anyway, but at least they'll get a useful error message.
            why_not_fast_path = f"dtypes of query ({query.dtype}) and self.in_proj_weight ({self.in_proj_weight.dtype}) don't match"
        elif self.training:
            why_not_fast_path = "training is enabled"
        elif not self.batch_first:
            why_not_fast_path = "batch_first was not True"
        elif self.bias_k is not None:
            why_not_fast_path = "self.bias_k was not None"
        elif self.bias_v is not None:
            why_not_fast_path = "self.bias_v was not None"
        elif self.dropout:
            why_not_fast_path = f"dropout was {self.dropout}, required zero"
        elif self.add_zero_attn:
            why_not_fast_path = "add_zero_attn was enabled"
        elif not self._qkv_same_embed_dim:
            why_not_fast_path = "_qkv_same_embed_dim was not True"
        elif attn_mask is not None:
            why_not_fast_path = "attn_mask was not None"
        elif query.is_nested and key_padding_mask is not None:
            why_not_fast_path = "key_padding_mask is not supported with NestedTensor input"
        elif self.num_heads % 2 == 1:
            why_not_fast_path = "num_heads is odd"
        elif torch.is_autocast_enabled():
            why_not_fast_path = "autocast is enabled"

        if not why_not_fast_path:
            tensor_args = (
                query,
                key,
                value,
                self.in_proj_weight,
                self.in_proj_bias,
                self.out_proj.weight,
                self.out_proj.bias,
            )
            # We have to use list comprehensions below because TorchScript does not support
            # generator expressions.
            if torch.overrides.has_torch_function(tensor_args):
                why_not_fast_path = "some Tensor argument has_torch_function"
            elif not all([(x is None or x.is_cuda or 'cpu' in str(x.device)) for x in tensor_args]):
                why_not_fast_path = "some Tensor argument is neither CUDA nor CPU"
            elif torch.is_grad_enabled() and any([x is not None and x.requires_grad for x in tensor_args]):
                why_not_fast_path = ("grad is enabled and at least one of query or the "
                                     "input/output projection weights or biases requires_grad")
            if not why_not_fast_path:
                return torch._native_multi_head_attention(
                    query,
                    key,
                    value,
                    self.embed_dim,
                    self.num_heads,
                    self.in_proj_weight,
                    self.in_proj_bias,
                    self.out_proj.weight,
                    self.out_proj.bias,
                    key_padding_mask if key_padding_mask is not None else attn_mask,
                    need_weights,
                    average_attn_weights,
                    1 if key_padding_mask is not None else 0 if attn_mask is not None else None)

        any_nested = query.is_nested or key.is_nested or value.is_nested
        assert not any_nested, ("MultiheadAttention does not support NestedTensor outside of its fast path. " +
                                f"The fast path was not hit because {why_not_fast_path}")

        if self.batch_first and is_batched:
            # make sure that the transpose op does not affect the "is" property
            if key is value:
                if query is key:
                    query = key = value = query.transpose(1, 0)
                else:
                    query, key = [x.transpose(1, 0) for x in (query, key)]
                    value = key
            else:
                query, key, value = [x.transpose(1, 0) for x in (query, key, value)]

        # self.init_incremental_state()
        if incremental_state is None:
            if not self._qkv_same_embed_dim:
                attn_output, attn_output_weights = F.multi_head_attention_forward(
                    query, key, value, self.embed_dim, self.num_heads,
                    self.in_proj_weight, self.in_proj_bias,
                    self.bias_k, self.bias_v, self.add_zero_attn,
                    self.dropout, self.out_proj.weight, self.out_proj.bias,
                    training=self.training,
                    key_padding_mask=key_padding_mask, need_weights=need_weights,
                    attn_mask=attn_mask, use_separate_proj_weight=True,
                    q_proj_weight=self.q_proj_weight, k_proj_weight=self.k_proj_weight,
                    v_proj_weight=self.v_proj_weight, average_attn_weights=average_attn_weights)
            else:
                attn_output, attn_output_weights = F.multi_head_attention_forward(
                    query, key, value, self.embed_dim, self.num_heads,
                    self.in_proj_weight, self.in_proj_bias,
                    self.bias_k, self.bias_v, self.add_zero_attn,
                    self.dropout, self.out_proj.weight, self.out_proj.bias,
                    training=self.training,
                    key_padding_mask=key_padding_mask, need_weights=need_weights,
                    attn_mask=attn_mask, average_attn_weights=average_attn_weights)
            if self.batch_first and is_batched:
                return attn_output.transpose(1, 0), attn_output_weights
            else:
                return attn_output, attn_output_weights
        
        # incremental_state is not None
        is_batched = _mha_shape_check(query, key, value, key_padding_mask, attn_mask, self.num_heads)

        # For unbatched input, we unsqueeze at the expected batch-dim to pretend that the input
        # is batched, run the computation and before returning squeeze the
        # batch dimension so that the output doesn't carry this temporary batch dimension.
        if not is_batched:
            # unsqueeze if the input is unbatched
            query = query.unsqueeze(1)
            key = key.unsqueeze(1)
            value = value.unsqueeze(1)
            if key_padding_mask is not None:
                key_padding_mask = key_padding_mask.unsqueeze(0)

        # set up shape vars
        tgt_len, bsz, embed_dim = query.shape
        src_len, _, _ = key.shape
        if key_padding_mask is not None:
            _kpm_dtype = key_padding_mask.dtype
            if _kpm_dtype != torch.bool and not torch.is_floating_point(key_padding_mask):
                raise AssertionError(
                    "only bool and floating types of key_padding_mask are supported")
        assert embed_dim == self.embed_dim, \
            f"was expecting embedding dimension of {self.embed_dim}, but got {embed_dim}"
        if isinstance(embed_dim, torch.Tensor):
            # embed_dim can be a tensor when JIT tracing
            head_dim = embed_dim.div(self.num_heads, rounding_mode='trunc')
        else:
            head_dim = embed_dim // self.num_heads
        assert head_dim * self.num_heads == embed_dim, f"embed_dim {embed_dim} not divisible by num_heads {self.num_heads}"
        
        if not self._qkv_same_embed_dim:
            # allow MHA to have different embedding dimensions when separate projection weights are used
            assert key.shape[:2] == value.shape[:2], \
                f"key's sequence and batch dims {key.shape[:2]} do not match value's {value.shape[:2]}"
        else:
            assert key.shape == value.shape, f"key shape {key.shape} does not match value shape {value.shape}"

        #------------------------------------------------------------------------------------
        # check if incremental_state exists
        if incremental_state is not None:
            # whether to use Incremental Decoding
            if "prev_key" in incremental_state.keys():
                # whether the first visit to it.
                saved_state = incremental_state
                assert saved_state is not None
            else:
                saved_state = None
        # NOTE: norm first?
        #------------------------------------------------------------------------------------

        #
        # compute in-projection
        #

        if self._qkv_same_embed_dim:
            assert self.in_proj_weight is not None, "use_separate_proj_weight is False but in_proj_weight is None"
            q, k, v = _in_projection_packed(query, key, value, self.in_proj_weight, self.in_proj_bias)
        else:
            assert self.q_proj_weight is not None, "use_separate_proj_weight is True but q_proj_weight is None"
            assert self.k_proj_weight is not None, "use_separate_proj_weight is True but k_proj_weight is None"
            assert self.v_proj_weight is not None, "use_separate_proj_weight is True but v_proj_weight is None"
            if self.in_proj_bias is None:
                b_q = b_k = b_v = None
            else:
                b_q, b_k, b_v = self.in_proj_bias.chunk(3)
            q, k, v = _in_projection(query, key, value, self.q_proj_weight, self.k_proj_weight, self.v_proj_weight, b_q, b_k, b_v)

        # # prep attention mask
        # if attn_mask is not None:
        #     if attn_mask.dtype == torch.uint8:
        #         warnings.warn("Byte tensor for attn_mask in nn.MultiheadAttention is deprecated. Use bool tensor instead.")
        #         attn_mask = attn_mask.to(torch.bool)
        #     else:
        #         assert attn_mask.is_floating_point() or attn_mask.dtype == torch.bool, \
        #             f"Only float, byte, and bool types are supported for attn_mask, not {attn_mask.dtype}"
        #     # ensure attn_mask's dim is 3
        #     if attn_mask.dim() == 2:
        #         correct_2d_size = (tgt_len, src_len)
        #         if attn_mask.shape != correct_2d_size:
        #             raise RuntimeError(f"The shape of the 2D attn_mask is {attn_mask.shape}, but should be {correct_2d_size}.")
        #         attn_mask = attn_mask.unsqueeze(0)
        #     elif attn_mask.dim() == 3:
        #         correct_3d_size = (bsz * self.num_heads, tgt_len, src_len)
        #         if attn_mask.shape != correct_3d_size:
        #             raise RuntimeError(f"The shape of the 3D attn_mask is {attn_mask.shape}, but should be {correct_3d_size}.")
        #     else:
        #         raise RuntimeError(f"attn_mask's dimension {attn_mask.dim()} is not supported")

        # add bias along batch dimension (currently second)
        if self.bias_k is not None:
            raise NotADirectoryError

        #
        # reshape q, k, v for multihead attention and make em batch first
        #
        q = q.contiguous().view(tgt_len, bsz * self.num_heads, head_dim).transpose(0, 1)
        kv_bsz = bsz

        # static KV not implemented
        k = k.contiguous().view(k.shape[0], bsz * self.num_heads, head_dim).transpose(0, 1)
        v = v.contiguous().view(v.shape[0], bsz * self.num_heads, head_dim).transpose(0, 1)
    
        
        # reshape the cached Key and Value:
        if incremental_state is not None:
            # Increnmental Decoding
            if saved_state is not None:
                # Not the first step
                if "prev_key" in saved_state:
                    _prev_key = saved_state["prev_key"]
                    assert _prev_key is not None
                    kv_bsz = _prev_key.size(0)
                    # Use contiguous() for reshaping
                    prev_key = _prev_key.contiguous().view(kv_bsz * self.num_heads, -1, head_dim)
                    assert k is not None
                    k = torch.cat([prev_key, k], dim=1)
                    src_len = k.size(1)
                if "prev_value" in saved_state:
                    _prev_value = saved_state["prev_value"]
                    assert _prev_value is not None
                    assert kv_bsz == _prev_value.size(0)
                    prev_value = _prev_value.contiguous().view(
                        kv_bsz * self.num_heads, -1, head_dim
                    )
                    assert v is not None
                    v = torch.cat([prev_value, v], dim=1)

                # _prev_value = saved_state["prev_value"]
            else:
                saved_state = {}
            saved_state["prev_key"] = k.view(kv_bsz, self.num_heads, -1, head_dim)
            saved_state["prev_value"] = v.view(kv_bsz, self.num_heads, -1, head_dim)
            assert k.size(0) == v.size(0), f"Key ({k.shape}) and value ({v.shape}) are not in the same shape." 
        else:
            # Not Incremental Decoding
            pass
        
        # Renew incremental state:
        # self.set_incremental_state(incremental_state, saved_state)
        self.set_incremental_state(
            incremental_state = incremental_state,
            saved_state = saved_state
        )

        # add zero attention along batch dimension (now first)
        if self.add_zero_attn:
            raise NotImplementedError

        # update source sequence length after adjustments
        src_len = k.size(1)

        # prep attention mask
        if attn_mask is not None:
            if attn_mask.dtype == torch.uint8:
                warnings.warn("Byte tensor for attn_mask in nn.MultiheadAttention is deprecated. Use bool tensor instead.")
                attn_mask = attn_mask.to(torch.bool)
            else:
                assert attn_mask.is_floating_point() or attn_mask.dtype == torch.bool, \
                    f"Only float, byte, and bool types are supported for attn_mask, not {attn_mask.dtype}"
            # ensure attn_mask's dim is 3
            if attn_mask.dim() == 2:
                correct_2d_size = (tgt_len, src_len)
                if attn_mask.shape != correct_2d_size:
                    raise RuntimeError(f"The shape of the 2D attn_mask is {attn_mask.shape}, but should be {correct_2d_size}.")
                attn_mask = attn_mask.unsqueeze(0)
            elif attn_mask.dim() == 3:
                correct_3d_size = (bsz * self.num_heads, tgt_len, src_len)
                if attn_mask.shape != correct_3d_size:
                    raise RuntimeError(f"The shape of the 3D attn_mask is {attn_mask.shape}, but should be {correct_3d_size}.")
            else:
                raise RuntimeError(f"attn_mask's dimension {attn_mask.dim()} is not supported")

        # merge key padding and attention masks
        if key_padding_mask is not None:
            assert key_padding_mask.shape == (bsz, src_len), \
                f"expecting key_padding_mask shape of {(bsz, src_len)}, but got {key_padding_mask.shape}"
            key_padding_mask = key_padding_mask.view(bsz, 1, 1, src_len).   \
                expand(-1, self.num_heads, -1, -1).reshape(bsz * self.num_heads, 1, src_len)
            if attn_mask is None:
                attn_mask = key_padding_mask
            elif attn_mask.dtype == torch.bool:
                attn_mask = attn_mask.logical_or(key_padding_mask)
            else:
                attn_mask = attn_mask.masked_fill(key_padding_mask, float("-inf"))

        # convert mask to float
        if attn_mask is not None and attn_mask.dtype == torch.bool:
            new_attn_mask = torch.zeros_like(attn_mask, dtype=q.dtype)
            new_attn_mask.masked_fill_(attn_mask, float("-inf"))
            attn_mask = new_attn_mask

        # adjust dropout probability
        if not self.training:
            dropout_p = 0.0

        #
        # (deep breath) calculate attention and out projection
        #

        B, Nt, E = q.shape
        q_scaled = q / math.sqrt(E)
        # attn_mask[:,-1:]
        if attn_mask is not None:
            attn_output_weights = torch.baddbmm(attn_mask, q_scaled, k.transpose(-2, -1))
        else:
            attn_output_weights = torch.bmm(q_scaled, k.transpose(-2, -1))
        attn_output_weights = softmax(attn_output_weights, dim=-1)
        if dropout_p > 0.0:
            attn_output_weights = dropout(attn_output_weights, p=dropout_p)

        attn_output = torch.bmm(attn_output_weights, v)

        attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len * bsz, embed_dim)
        attn_output = linear(attn_output, self.out_proj.weight, self.out_proj.bias)
        attn_output = attn_output.view(tgt_len, bsz, attn_output.size(1))

        if need_weights:
            # optionally average attention weights over heads
            attn_output_weights = attn_output_weights.view(bsz, self.num_heads, tgt_len, src_len)
            if average_attn_weights:
                attn_output_weights = attn_output_weights.sum(dim=1) / self.num_heads

            if not is_batched:
                # squeeze the output if input was unbatched
                attn_output = attn_output.squeeze(1)
                attn_output_weights = attn_output_weights.squeeze(0)
            # return attn_output, attn_output_weights
        else:
            if not is_batched:
                # squeeze the output if input was unbatched
                attn_output = attn_output.squeeze(1)
                attn_output_weights = None
            # return attn_output, None
        if self.batch_first and is_batched:
            return attn_output.transpose(1, 0), attn_output_weights
        else:
            return attn_output, attn_output_weights


def multi_head_attention_forward_incremental(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    incremental_state: None,
    embed_dim_to_check: int,
    num_heads: int,
    in_proj_weight: Optional[Tensor],
    in_proj_bias: Optional[Tensor],
    bias_k: Optional[Tensor],
    bias_v: Optional[Tensor],
    add_zero_attn: bool,
    dropout_p: float,
    out_proj_weight: Tensor,
    out_proj_bias: Optional[Tensor],
    training: bool = True,
    key_padding_mask: Optional[Tensor] = None,
    need_weights: bool = True,
    attn_mask: Optional[Tensor] = None,
    use_separate_proj_weight: bool = False,
    q_proj_weight: Optional[Tensor] = None,
    k_proj_weight: Optional[Tensor] = None,
    v_proj_weight: Optional[Tensor] = None,
    static_k: Optional[Tensor] = None,
    static_v: Optional[Tensor] = None,
    average_attn_weights: bool = True,
) -> Tuple[Tensor, Optional[Tensor]]:
    r"""
    Args:
        query, key, value: map a query and a set of key-value pairs to an output.
            See "Attention Is All You Need" for more details.
        embed_dim_to_check: total dimension of the model.
        num_heads: parallel attention heads.
        in_proj_weight, in_proj_bias: input projection weight and bias.
        bias_k, bias_v: bias of the key and value sequences to be added at dim=0.
        add_zero_attn: add a new batch of zeros to the key and
                       value sequences at dim=1.
        dropout_p: probability of an element to be zeroed.
        out_proj_weight, out_proj_bias: the output projection weight and bias.
        training: apply dropout if is ``True``.
        key_padding_mask: if provided, specified padding elements in the key will
            be ignored by the attention. This is an binary mask. When the value is True,
            the corresponding value on the attention layer will be filled with -inf.
        need_weights: output attn_output_weights.
        attn_mask: 2D or 3D mask that prevents attention to certain positions. A 2D mask will be broadcasted for all
            the batches while a 3D mask allows to specify a different mask for the entries of each batch.
        use_separate_proj_weight: the function accept the proj. weights for query, key,
            and value in different forms. If false, in_proj_weight will be used, which is
            a combination of q_proj_weight, k_proj_weight, v_proj_weight.
        q_proj_weight, k_proj_weight, v_proj_weight, in_proj_bias: input projection weight and bias.
        static_k, static_v: static key and value used for attention operators.
        average_attn_weights: If true, indicates that the returned ``attn_weights`` should be averaged across heads.
            Otherwise, ``attn_weights`` are provided separately per head. Note that this flag only has an effect
            when ``need_weights=True.``. Default: True


    Shape:
        Inputs:
        - query: :math:`(L, E)` or :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
          the embedding dimension.
        - key: :math:`(S, E)` or :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - value: :math:`(S, E)` or :math:`(S, N, E)` where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - key_padding_mask: :math:`(S)` or :math:`(N, S)` where N is the batch size, S is the source sequence length.
          If a FloatTensor is provided, it will be directly added to the value.
          If a BoolTensor is provided, the positions with the
          value of ``True`` will be ignored while the position with the value of ``False`` will be unchanged.
        - attn_mask: 2D mask :math:`(L, S)` where L is the target sequence length, S is the source sequence length.
          3D mask :math:`(N*num_heads, L, S)` where N is the batch size, L is the target sequence length,
          S is the source sequence length. attn_mask ensures that position i is allowed to attend the unmasked
          positions. If a ByteTensor is provided, the non-zero positions are not allowed to attend
          while the zero positions will be unchanged. If a BoolTensor is provided, positions with ``True``
          are not allowed to attend while ``False`` values will be unchanged. If a FloatTensor
          is provided, it will be added to the attention weight.
        - static_k: :math:`(N*num_heads, S, E/num_heads)`, where S is the source sequence length,
          N is the batch size, E is the embedding dimension. E/num_heads is the head dimension.
        - static_v: :math:`(N*num_heads, S, E/num_heads)`, where S is the source sequence length,
          N is the batch size, E is the embedding dimension. E/num_heads is the head dimension.

        Outputs:
        - attn_output: :math:`(L, E)` or :math:`(L, N, E)` where L is the target sequence length, N is the batch size,
          E is the embedding dimension.
        - attn_output_weights: Only returned when ``need_weights=True``. If ``average_attn_weights=True``, returns
          attention weights averaged across heads of shape :math:`(L, S)` when input is unbatched or
          :math:`(N, L, S)`, where :math:`N` is the batch size, :math:`L` is the target sequence length, and
          :math:`S` is the source sequence length. If ``average_attn_weights=False``, returns attention weights per
          head of shape :math:`(num_heads, L, S)` when input is unbatched or :math:`(N, num_heads, L, S)`.
    """
    tens_ops = (query, key, value, in_proj_weight, in_proj_bias, bias_k, bias_v, out_proj_weight, out_proj_bias)
    if has_torch_function(tens_ops):
        return handle_torch_function(
            multi_head_attention_forward_incremental,
            tens_ops,
            query,
            key,
            incremental_state,
            value,
            embed_dim_to_check,
            num_heads,
            in_proj_weight,
            in_proj_bias,
            bias_k,
            bias_v,
            add_zero_attn,
            dropout_p,
            out_proj_weight,
            out_proj_bias,
            training=training,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            attn_mask=attn_mask,
            use_separate_proj_weight=use_separate_proj_weight,
            q_proj_weight=q_proj_weight,
            k_proj_weight=k_proj_weight,
            v_proj_weight=v_proj_weight,
            static_k=static_k,
            static_v=static_v,
            average_attn_weights=average_attn_weights,
        )

    is_batched = _mha_shape_check(query, key, value, key_padding_mask, attn_mask, num_heads)

    # For unbatched input, we unsqueeze at the expected batch-dim to pretend that the input
    # is batched, run the computation and before returning squeeze the
    # batch dimension so that the output doesn't carry this temporary batch dimension.
    if not is_batched:
        # unsqueeze if the input is unbatched
        query = query.unsqueeze(1)
        key = key.unsqueeze(1)
        value = value.unsqueeze(1)
        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.unsqueeze(0)

    # set up shape vars
    tgt_len, bsz, embed_dim = query.shape
    src_len, _, _ = key.shape
    if key_padding_mask is not None:
        _kpm_dtype = key_padding_mask.dtype
        if _kpm_dtype != torch.bool and not torch.is_floating_point(key_padding_mask):
            raise AssertionError(
                "only bool and floating types of key_padding_mask are supported")
    assert embed_dim == embed_dim_to_check, \
        f"was expecting embedding dimension of {embed_dim_to_check}, but got {embed_dim}"
    if isinstance(embed_dim, torch.Tensor):
        # embed_dim can be a tensor when JIT tracing
        head_dim = embed_dim.div(num_heads, rounding_mode='trunc')
    else:
        head_dim = embed_dim // num_heads
    assert head_dim * num_heads == embed_dim, f"embed_dim {embed_dim} not divisible by num_heads {num_heads}"
    if use_separate_proj_weight:
        # allow MHA to have different embedding dimensions when separate projection weights are used
        assert key.shape[:2] == value.shape[:2], \
            f"key's sequence and batch dims {key.shape[:2]} do not match value's {value.shape[:2]}"
    else:
        assert key.shape == value.shape, f"key shape {key.shape} does not match value shape {value.shape}"

    #------------------------------------------------------------------------------------
    # check if incremental_state exists
    if incremental_state is not None:
        # whether to use Incremental Decoding
        if "prev_key" in incremental_state.keys():
            # whether the first visit to it.
            saved_state = incremental_state
            assert saved_state is not None
        else:
            saved_state = None
    # NOTE: norm first?
    #------------------------------------------------------------------------------------

    #
    # compute in-projection
    #

    if not use_separate_proj_weight:
        assert in_proj_weight is not None, "use_separate_proj_weight is False but in_proj_weight is None"
        q, k, v = _in_projection_packed(query, key, value, in_proj_weight, in_proj_bias)
    else:
        assert q_proj_weight is not None, "use_separate_proj_weight is True but q_proj_weight is None"
        assert k_proj_weight is not None, "use_separate_proj_weight is True but k_proj_weight is None"
        assert v_proj_weight is not None, "use_separate_proj_weight is True but v_proj_weight is None"
        if in_proj_bias is None:
            b_q = b_k = b_v = None
        else:
            b_q, b_k, b_v = in_proj_bias.chunk(3)
        q, k, v = _in_projection(query, key, value, q_proj_weight, k_proj_weight, v_proj_weight, b_q, b_k, b_v)

    # prep attention mask
    if attn_mask is not None:
        if attn_mask.dtype == torch.uint8:
            warnings.warn("Byte tensor for attn_mask in nn.MultiheadAttention is deprecated. Use bool tensor instead.")
            attn_mask = attn_mask.to(torch.bool)
        else:
            assert attn_mask.is_floating_point() or attn_mask.dtype == torch.bool, \
                f"Only float, byte, and bool types are supported for attn_mask, not {attn_mask.dtype}"
        # ensure attn_mask's dim is 3
        if attn_mask.dim() == 2:
            correct_2d_size = (tgt_len, src_len)
            if attn_mask.shape != correct_2d_size:
                raise RuntimeError(f"The shape of the 2D attn_mask is {attn_mask.shape}, but should be {correct_2d_size}.")
            attn_mask = attn_mask.unsqueeze(0)
        elif attn_mask.dim() == 3:
            correct_3d_size = (bsz * num_heads, tgt_len, src_len)
            if attn_mask.shape != correct_3d_size:
                raise RuntimeError(f"The shape of the 3D attn_mask is {attn_mask.shape}, but should be {correct_3d_size}.")
        else:
            raise RuntimeError(f"attn_mask's dimension {attn_mask.dim()} is not supported")

    # add bias along batch dimension (currently second)
    if bias_k is not None and bias_v is not None:
        assert static_k is None, "bias cannot be added to static key."
        assert static_v is None, "bias cannot be added to static value."
        k = torch.cat([k, bias_k.repeat(1, bsz, 1)])
        v = torch.cat([v, bias_v.repeat(1, bsz, 1)])
        if attn_mask is not None:
            attn_mask = pad(attn_mask, (0, 1))
        if key_padding_mask is not None:
            key_padding_mask = pad(key_padding_mask, (0, 1))
    else:
        assert bias_k is None
        assert bias_v is None

    #
    # reshape q, k, v for multihead attention and make em batch first
    #
    q = q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
    kv_bsz = bsz
    if static_k is None:
        k = k.contiguous().view(k.shape[0], bsz * num_heads, head_dim).transpose(0, 1)
    else:
        # TODO finish disentangling control flow so we don't do in-projections when statics are passed
        assert static_k.size(0) == bsz * num_heads, \
            f"expecting static_k.size(0) of {bsz * num_heads}, but got {static_k.size(0)}"
        assert static_k.size(2) == head_dim, \
            f"expecting static_k.size(2) of {head_dim}, but got {static_k.size(2)}"
        k = static_k
    if static_v is None:
        v = v.contiguous().view(v.shape[0], bsz * num_heads, head_dim).transpose(0, 1)
    else:
        # TODO finish disentangling control flow so we don't do in-projections when statics are passed
        assert static_v.size(0) == bsz * num_heads, \
            f"expecting static_v.size(0) of {bsz * num_heads}, but got {static_v.size(0)}"
        assert static_v.size(2) == head_dim, \
            f"expecting static_v.size(2) of {head_dim}, but got {static_v.size(2)}"
        v = static_v
    
    # reshape the cached Key and Value:
    if incremental_state is not None:
        # Increnmental Decoding
        if saved_state is not None:
            # Not the first step
            if "prev_key" in saved_state:
                _prev_key = saved_state["prev_key"]
                assert _prev_key is not None
                kv_bsz = _prev_key.size(0)
                prev_key = _prev_key.view(kv_bsz * num_heads, -1, head_dim)
                if static_k:
                    k = prev_key
                else:
                    assert k is not None
                    k = torch.cat([prev_key, k], dim=1)
                src_len = k.size(1)
            if "prev_value" in saved_state:
                _prev_value = saved_state["prev_value"]
                assert _prev_value is not None
                assert kv_bsz == _prev_value.size(0)
                prev_value = _prev_value.view(
                    kv_bsz * num_heads, -1, head_dim
                )
                if static_v:
                    v = prev_value
                else:
                    assert v is not None
                    v = torch.cat([prev_value, v], dim=1)

            # _prev_value = saved_state["prev_value"]
        else:
            saved_state = {}
        saved_state["prev_key"] = k.view(kv_bsz, num_heads, -1, head_dim)
        saved_state["prev_value"] = v.view(kv_bsz, num_heads, -1, head_dim)
        assert k.size(0) == v.size(0), f"Key ({k.shape}) and value ({v.shape}) are not in the same shape." 
    else:
        # Not Incremental Decoding
        pass

    # add zero attention along batch dimension (now first)
    if add_zero_attn:
        zero_attn_shape = (bsz * num_heads, 1, head_dim)
        k = torch.cat([k, torch.zeros(zero_attn_shape, dtype=k.dtype, device=k.device)], dim=1)
        v = torch.cat([v, torch.zeros(zero_attn_shape, dtype=v.dtype, device=v.device)], dim=1)
        if attn_mask is not None:
            attn_mask = pad(attn_mask, (0, 1))
        if key_padding_mask is not None:
            key_padding_mask = pad(key_padding_mask, (0, 1))

    # update source sequence length after adjustments
    src_len = k.size(1)

    # merge key padding and attention masks
    if key_padding_mask is not None:
        assert key_padding_mask.shape == (bsz, src_len), \
            f"expecting key_padding_mask shape of {(bsz, src_len)}, but got {key_padding_mask.shape}"
        key_padding_mask = key_padding_mask.view(bsz, 1, 1, src_len).   \
            expand(-1, num_heads, -1, -1).reshape(bsz * num_heads, 1, src_len)
        if attn_mask is None:
            attn_mask = key_padding_mask
        elif attn_mask.dtype == torch.bool:
            attn_mask = attn_mask.logical_or(key_padding_mask)
        else:
            attn_mask = attn_mask.masked_fill(key_padding_mask, float("-inf"))

    # convert mask to float
    if attn_mask is not None and attn_mask.dtype == torch.bool:
        new_attn_mask = torch.zeros_like(attn_mask, dtype=q.dtype)
        new_attn_mask.masked_fill_(attn_mask, float("-inf"))
        attn_mask = new_attn_mask

    # adjust dropout probability
    if not training:
        dropout_p = 0.0

    #
    # (deep breath) calculate attention and out projection
    #

    B, Nt, E = q.shape
    q_scaled = q / math.sqrt(E)
    # attn_mask[:,-1:]
    if attn_mask is not None:
        attn_output_weights = torch.baddbmm(attn_mask, q_scaled, k.transpose(-2, -1))
    else:
        attn_output_weights = torch.bmm(q_scaled, k.transpose(-2, -1))
    attn_output_weights = softmax(attn_output_weights, dim=-1)
    if dropout_p > 0.0:
        attn_output_weights = dropout(attn_output_weights, p=dropout_p)

    attn_output = torch.bmm(attn_output_weights, v)

    attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len * bsz, embed_dim)
    attn_output = linear(attn_output, out_proj_weight, out_proj_bias)
    attn_output = attn_output.view(tgt_len, bsz, attn_output.size(1))

    if need_weights:
        # optionally average attention weights over heads
        attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
        if average_attn_weights:
            attn_output_weights = attn_output_weights.sum(dim=1) / num_heads

        if not is_batched:
            # squeeze the output if input was unbatched
            attn_output = attn_output.squeeze(1)
            attn_output_weights = attn_output_weights.squeeze(0)
        return attn_output, attn_output_weights
    else:
        if not is_batched:
            # squeeze the output if input was unbatched
            attn_output = attn_output.squeeze(1)
        return attn_output, None
