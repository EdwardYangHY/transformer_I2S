import sys
import math

import torch
from torch import nn
import torch.nn.functional as F
# from PureT_encoder import Encoder as refine_encoder
import uuid
from typing import Optional, Any, Union, Callable
from torch import Tensor
from incremental_utils import with_incremental_state

from torch.nn.modules.activation import MultiheadAttention
from incremental_mha import MultiheadAttention_incremental

# from models_prompt import prefix_TransformerEncoder, TransformerPrefixLM

@with_incremental_state
class incremental_TransformerEncoderLayer(nn.TransformerEncoderLayer):
    def __init__(
        self, 
        d_model: int, 
        nhead: int, 
        dim_feedforward: int = 2048, 
        dropout: float = 0.1, 
        activation: Union[str, Callable[[Tensor], Tensor]] = F.relu, 
        layer_norm_eps: float = 0.00001, 
        batch_first: bool = False, 
        norm_first: bool = False, 
        device=None, 
        dtype=None,
        ) -> None:
        super().__init__(
            d_model, 
            nhead, 
            dim_feedforward, 
            dropout, 
            activation, 
            layer_norm_eps, 
            batch_first, 
            norm_first, 
            device, 
            dtype
        )
        self.d_model = d_model
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.self_attn = MultiheadAttention_incremental(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                            **factory_kwargs)
        # self.incremental_state = None
    # def set_layer_id(self):
    #     self._layer_id = str(uuid.uuid4())

    def forward(
            self, 
            src: Tensor,
            key: Tensor,
            value: Tensor,
            incremental_state: None,
            src_mask: Optional[Tensor] = None,
            src_key_padding_mask: Optional[Tensor] = None
        ) -> Tensor:
        """
            Customized Encoder Layer.
            key_prompt: Prefix-prompt key. It should have the same size as "src"
            value_prompt: Prefix-prompt value. It should have the same size as "src"
            For e.g.:
                The original seq length is T; the prefix length is T'.
                Concaten seq length should be T'+T, while the [B, :T', dim] is promptable, which
                means, controlled by params.
                And in each layer of Transformer, the QKV is shaped like [B, T'+T, dim], so the 
                prefix part should also be promptable.

            If key_prompt == value_prompt == x, this equals to a standard TransformerEncoderLayer.
        """
        x = src
        assert x.shape == key.shape == value.shape, f"Q {x.shape}, K {key.shape}, V {value.shape} should have the same size"
        if self.norm_first:
            # x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask)
            x = x + self._sa_block_incremental(
                q = self.norm1(x), 
                k = self.norm1(key), 
                v = self.norm1(value), 
                incremental_state = incremental_state,
                attn_mask = src_mask, 
                key_padding_mask = src_key_padding_mask
            ) 
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block_incremental(x, key, value, incremental_state, src_mask, src_key_padding_mask))
            x = self.norm2(x + self._ff_block(x))
        return x

    # def _sa_block_incremental(
    #         self,
    #         q: Tensor,
    #         k: Tensor,
    #         v: Tensor,
    #         incremental_state: None,
    #         attn_mask: Optional[Tensor], 
    #         key_padding_mask: Optional[Tensor]
    #     ) -> Tensor:
    #     """ 
    #     Customized Self-attention Block for prefix prompt tuning.
    #     Ref: Prefix-tuning https://arxiv.org/abs/2101.00190

    #     originally, (..., need_weights=Flase)[0] outputs no atten.
    #     To
    #     """
    #     # _layer_id would not be changed
    #     if incremental_state != None:
    #         assert self._layer_id is not None, "Incremental decoding can't work without layer id"
    #         self.init_incremental_state(incremental_state, self._layer_id)
    #         layer_incremental_state = self.get_incremental_state(incremental_state, self._layer_id)
    #     else:
    #         layer_incremental_state = None
        
    #     x, atten = self.self_attn(query=q, key=k, value=v,
    #                         incremental_state=layer_incremental_state,
    #                        attn_mask=attn_mask,
    #                        key_padding_mask=key_padding_mask,
    #                        need_weights=True)# [0]
        
    #     return self.dropout1(x)
    
@with_incremental_state
class incremental_TransformerDecoderLayer(nn.TransformerDecoderLayer):
    def __init__(
        self, 
        d_model: int, 
        nhead: int, 
        dim_feedforward: int = 2048, 
        dropout: float = 0.1, 
        activation: Union[str, Callable[[Tensor], Tensor]] = F.relu, 
        layer_norm_eps: float = 0.00001, 
        batch_first: bool = False, 
        norm_first: bool = False, 
        tau: float = 0.2,
        device=None, 
        dtype=None,
        ) -> None:
        super().__init__(
            d_model, 
            nhead, 
            dim_feedforward, 
            dropout, 
            activation, 
            layer_norm_eps, 
            batch_first, 
            norm_first, 
            device, 
            dtype
        )
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.self_attn = MultiheadAttention_incremental(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                            **factory_kwargs)
    
    def forward(
        self, 
        tgt: Tensor, 
        memory: Tensor, 
        incremental_state: None,
        tgt_mask: Optional[Tensor] = None, 
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None, 
        memory_key_padding_mask: Optional[Tensor] = None
        ) -> Tensor:

        x = tgt
        key = tgt
        value = tgt
        """
            Cross Attention does not need incremental decoding, because memory is always computed
            in mha module.
            And memory can be shaped differently, so we direclty use previous mha module
        """
        if self.norm_first:
            # x = x + self._sa_block_incremental(self.norm1(x), tgt_mask, tgt_key_padding_mask)
            x = x + self._sa_block_incremental(
                q = self.norm1(x), 
                k = self.norm1(key), 
                v = self.norm1(value), 
                incremental_state = incremental_state,
                attn_mask = tgt_mask, 
                key_padding_mask = tgt_key_padding_mask
            ) 
            x = x + self._mha_block(self.norm2(x), memory, memory_mask, memory_key_padding_mask)
            x = x + self._ff_block(self.norm3(x))
        else:
            x = self.norm1(x + self._sa_block_incremental(x, key, value, incremental_state, src_mask, src_key_padding_mask))
            x = self.norm2(x + self._mha_block(x, memory, memory_mask, memory_key_padding_mask))
            x = self.norm3(x + self._ff_block(x))
        
        return x


# def beam_search(self, imgs, gx, beam_size, start_unit, end_unit, max_len, device):

#         incremental_state = {}
#         # incremental_state = None

#         k = beam_size
            
#         # Tensor to store top k previous words at each step; now they're just <start>
#         k_prev_words = torch.LongTensor([[start_unit]] * k).to(device)  # (k, 1)
#         # Tensor to store top k sequences; now they're just <start>
#         seqs = k_prev_words  # (k, 1)
#         # Tensor to store top k sequences' scores; now they're just 0
#         top_k_scores = torch.zeros(k, 1).to(device)  # (k, 1)
#         # Lists to store completed sequences and scores
#         complete_seqs = list()
#         complete_seqs_scores = list()
#         # Start decoding
#         step = 1
#         # s is a number less than or equal to k, because sequences are removed from this process once they hit <end>
#         self.LM_decoder.set_prefix(gx)

#         while True:
#             # first version
#             # gx = gx_origin.expand(k, 1, imgs.size(-1))
#             # imgs = imgs_origin.expand(k, imgs.size(-2), imgs.size(-1))

#             """
#                 Start from here. if the incremental_state is not None, then only embed the latest.
#             """
#             if incremental_state != None and step != 1:
#                 x = self.embed(seqs[:,-1:])
#                 x = self.pos_encoder(x, pos=step-1)
#                 decoder_input_attn_mask = self.generate_VALLE_mask(img_len=self.prefix_length, seq_len=step).to(x.device)
#                 attn_mask = decoder_input_attn_mask[-1:]
#                 pad_mask = None
#             else:
#                 x = self.embed(seqs)  # (1, seq, d_model)
#                 x = self.pos_encoder(x)  # (seq, 1, d_model)
#                 x, pad_mask, attn_mask = self.prefuse_gx(x, gx = gx, seq_padding_mask = None)
            
#             x = self.LM_decoder(
#                 src = x, 
#                 incremental_state = incremental_state,
#                 mask = attn_mask, 
#                 src_key_padding_mask = pad_mask
#                 )

#             scores = self.classifier(x[:, -1, :])  # (1, vocab_size)
#             scores = F.log_softmax(scores, dim=1)
#             # Add
#             scores = top_k_scores.expand_as(scores) + scores  # (s, vocab_size)
#             # For the first step, all k points will have the same scores (since same k previous words, h, c)
#             if step == 1:
#                 top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)  # (s)
#             else:
#                 # Unroll and find top scores, and their unrolled indices
#                 top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)  # (s)
#             # Convert unrolled indices to actual indices of scores
#             """
#                 prev_word_inds is the selected, new order of prev_seqs?
#             """
#             prev_word_inds = torch.div(top_k_words, self.vocab_size, rounding_mode="floor")  # (s)
#             next_word_inds = top_k_words % self.vocab_size  # (s)
#             # Add new words to sequences
#             seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)  # (s, step+1)
#             # Which sequences are incomplete (didn't reach <end>)?
#             incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if next_word != end_unit]
#             complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))
#             # Set aside complete sequences
#             if len(complete_inds) > 0:
#                 complete_seqs.extend(seqs[complete_inds].tolist())
#                 complete_seqs_scores.extend(top_k_scores[complete_inds])
#             k -= len(complete_inds)  # reduce beam length accordingly
#             # Proceed with incomplete sequences
#             if k == 0:
#                 break
#             seqs = seqs[incomplete_inds]
#             # imgs = imgs[prev_word_inds[incomplete_inds]]
#             """
#                 Use prev_word_inds[incomplete_inds] to sort the keys and values.
#             """
#             self.reoder_incremental_state(incremental_state, prev_word_inds[incomplete_inds])
#             gx = gx[prev_word_inds[incomplete_inds]]
#             top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
#             k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)
#             # Break if things have been going on too long
#             # print(k, step)
#             # print(seqs)
#             if step > max_len:
#                 break
#             step += 1
            
#         # print(seqs)
#         if len(complete_seqs_scores) != 0:
#             i = complete_seqs_scores.index(max(complete_seqs_scores))
#             seq = complete_seqs[i]
#         else:
#             seq = []

#         return seq