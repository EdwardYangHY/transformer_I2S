"""
BSD 3-Clause License
Copyright (c) 2017-2022, Pytorch contributors
All rights reserved.
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.
* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.
* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

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

from models_prompt import prefix_TransformerEncoder, TransformerPrefixLM

@with_incremental_state
class incremental_prefix_TransformerEncoderLayer(nn.TransformerEncoderLayer):
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
    def set_layer_id(self):
        self._layer_id = str(uuid.uuid4())

    def forward(
            self, 
            src: Tensor,
            key_prompt: Tensor,
            value_prompt: Tensor,
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
        assert x.shape == key_prompt.shape == value_prompt.shape, f"Q {x.shape}, K {key_prompt.shape}, V {value_prompt.shape} should have the same size"
        if self.norm_first:
            # x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask)
            x = x + self._sa_block_prompt(
                x = self.norm1(x), 
                k_prompt = self.norm1(key_prompt), 
                v_prompt = self.norm1(value_prompt), 
                incremental_state = incremental_state,
                attn_mask = src_mask, 
                key_padding_mask = src_key_padding_mask
            ) 
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block_prompt(x, key_prompt, value_prompt, incremental_state, src_mask, src_key_padding_mask))
            x = self.norm2(x + self._ff_block(x))
        return x

    def _sa_block_prompt(
            self,
            x: Tensor,
            k_prompt: Tensor,
            v_prompt: Tensor,
            incremental_state: None,
            attn_mask: Optional[Tensor], 
            key_padding_mask: Optional[Tensor]
        ) -> Tensor:
        """ 
        Customized Self-attention Block for prefix prompt tuning.
        Ref: Prefix-tuning https://arxiv.org/abs/2101.00190

        originally, (..., need_weights=Flase)[0] outputs no atten.
        To
        """
        # _layer_id would not be changed
        if incremental_state != None:
            # if self._layer_id not in incremental_state.keys():
            #     incremental_state[self._layer_id] = {}
            # layer_incremental_state = incremental_state[self._layer_id]
            self.init_incremental_state(incremental_state, self._layer_id)
            layer_incremental_state = self.get_incremental_state(incremental_state, self._layer_id)
        else:
            layer_incremental_state = None

        if len(layer_incremental_state.keys()) != 0:
            # not the first step of decoding.
            # all needed information was cached
            x_incremental, atten = self.self_attn(query=x[:,-1:], key=k_prompt[:,-1:], value=v_prompt[:,-1:],
                            incremental_state=layer_incremental_state,
                           attn_mask=attn_mask[-1:],
                           key_padding_mask=key_padding_mask,
                           need_weights=True)# [0]
        else:
            x_incremental, atten = self.self_attn(query=x, key=k_prompt, value=v_prompt,
                            incremental_state=layer_incremental_state,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=True)# [0]
        
        # a plain self_attention with no incremental state, for comparison
        x, atten = self.self_attn(query=x, key=k_prompt, value=v_prompt,
                            incremental_state=None,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=True)# [0]
        
        return self.dropout1(x)

class incremental_prefix_TransformerEncoder(prefix_TransformerEncoder):
    def __init__(
            self, 
            encoder_layer,
            num_layers, 
            prefix_length,
            norm=None, 
            enable_nested_tensor=True, 
            mask_check=True
            ):
        super().__init__(
            encoder_layer, 
            num_layers, 
            prefix_length,
            norm, 
            enable_nested_tensor, 
            mask_check
        )
        # self.d_model = self.layers[0].d_model
        # self.input_prompt_embedder = nn.Conv1d(in_channels=self.d_model, out_channels=self.d_model, kernel_size=1)
        # self.key_embedders = nn.ModuleList([nn.Conv1d(in_channels=self.d_model, out_channels=self.d_model, kernel_size=1) for i in range(num_layers)])
        # self.value_embedders = nn.ModuleList([nn.Conv1d(in_channels=self.d_model, out_channels=self.d_model, kernel_size=1) for i in range(num_layers)])
        # # # --------------------------------------------------------------------------------------------------------
        # # self.key_embedders = nn.ModuleList([nn.Linear(self.d_model, self.d_model) for i in range(num_layers)])
        # # self.value_embedders = nn.ModuleList([nn.Linear(self.d_model, self.d_model) for i in range(num_layers)])
        # # # --------------------------------------------------------------------------------------------------------
        # self.prompt_activation = nn.ReLU()
        # self.prompt_dropout = nn.Dropout(p=0.05)
        # self.prefix_length = prefix_length
        # self.prefix_cache = None
        for layer in self.layers:
            layer.set_layer_id()

        first_layer = self.layers[0]
        assert isinstance(first_layer, incremental_prefix_TransformerEncoderLayer), f"{first_layer} is not a incremental_prefix_TransformerEncoderLayer"

    def forward(
            self, 
            src: Tensor, 
            # prefix_embedders: None,
            incremental_state: None,
            mask: Optional[Tensor] = None, 
            src_key_padding_mask: Optional[Tensor] = None
        ) -> Tensor:
        """
            Here we assume that, like the parent class:
            src is shaped: [B, prefix_length + T, 1024]
            and key_padding_mask is already prepared.

            And note that, for Backward Propogation: 
            The promptable prefix of each layer does not from from the previous layer;
            instead it comes from the original prefix.
        """
        if self.training:
            # ignore incremental state if training.
            incremental_state = None
        
        output = src
        batch_size = output.size(0)
        if self.prefix_cache is not None:
            input_prefix = self.prefix_cache["prefix"][:batch_size,:,:]
        else:
            input_prefix = src.clone()
            input_prefix = input_prefix[:,:self.prefix_length,:]
        # first_layer = self.layers[0]
        # assert isinstance(first_layer, prefix_TransformerEncoderLayer), f"{first_layer} is not a prefix_TransformerEncoderLayer"
        
        """
            Before go into transformerlayers, we prompt the input prefix:
        """
        if self.prefix_cache is not None:
            input_prefix_prompt = self.prefix_cache["input_prompt"][:batch_size,:,:]
        else:
            input_prefix_prompt = self.input_prompt_embedder(input_prefix.permute(0,2,1)).permute(0,2,1)
            input_prefix_prompt = self.prompt_dropout(self.prompt_activation(input_prefix_prompt))

        output[:,:self.prefix_length,:] = input_prefix_prompt
        
        
        for idx, mod in enumerate(self.layers):
            key_prompt = output.clone()
            if self.prefix_cache is not None:
                key_prefix_prompt = self.prefix_cache["key_prompt"][idx][:batch_size,:,:]
            else:
                key_prefix_prompt = self.key_embedders[idx](input_prefix.permute(0,2,1)).permute(0,2,1)
                key_prefix_prompt = self.prompt_dropout(self.prompt_activation(key_prefix_prompt))
            key_prompt[:,:self.prefix_length,:] = key_prefix_prompt
            
            value_prompt = output.clone()
            if self.prefix_cache is not None:
                value_prefix_prompt = self.prefix_cache["value_prompt"][idx][:batch_size,:,:]
            else:
                value_prefix_prompt = self.value_embedders[idx](input_prefix.permute(0,2,1)).permute(0,2,1)
                value_prefix_prompt = self.prompt_dropout(self.prompt_activation(value_prefix_prompt))
            value_prompt[:,:self.prefix_length,:] = value_prefix_prompt

            output = mod(
                output, 
                key_prompt=key_prompt,
                value_prompt=value_prompt,
                incremental_state=incremental_state,
                src_mask=mask, 
                src_key_padding_mask=src_key_padding_mask
                )

        if self.norm is not None:
            output = self.norm(output)

        return output

class incremental_prefix_Transformer(TransformerPrefixLM):
    """
        Here we use a different way of LM.
        Previous conditioned LM is based on Transformer Decoder.
        Here, the LM is conditioned on prefix of Image Features, so it's based on Encoder Structure
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 1024,
        nhead: int = 8,
        num_layers: int = 6,
        activation="gelu",
        layer_norm_eps: float = 1e-5,
        batch_first: bool = True,
        norm_first: bool = True,
        dropout: float = 0.1,
        image_backbone: str = "ResNet",
        use_sentence_encoder: bool=False,
        sentence_embed: int = 0,
        fine_tune_image_encoder: bool = False,
        use_refine_encoder: bool = False,
        use_global_feature: bool = False,
        AR: bool = True,
        refine_encoder_params: dict = None
        ):
        super().__init__(
            vocab_size,
            d_model,
            nhead,
            num_layers,
            activation,
            layer_norm_eps,
            batch_first,
            norm_first,
            dropout,
            image_backbone,
            fine_tune_image_encoder,
            use_refine_encoder,
            use_global_feature,
            AR,
            refine_encoder_params
        )
        # if use_global_feature:
        #      self.prefix_length = refine_encoder_params["input_resolution"]^2 + 1
        # else:
        #      self.prefix_length = refine_encoder_params["input_resolution"]^2
        
        decoder_norm = nn.LayerNorm(d_model, eps=layer_norm_eps)
        decoder_layer = incremental_prefix_TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=4*d_model, dropout= self.dropout,
                                                   activation=activation, batch_first=batch_first, norm_first=norm_first)
        self.LM_decoder = incremental_prefix_TransformerEncoder(decoder_layer, num_layers, self.prefix_length, decoder_norm)


    def beam_search(self, imgs, gx, beam_size, start_unit, end_unit, max_len, device):

        incremental_state = {}

        k = beam_size
            
        # Tensor to store top k previous words at each step; now they're just <start>
        k_prev_words = torch.LongTensor([[start_unit]] * k).to(device)  # (k, 1)
        # Tensor to store top k sequences; now they're just <start>
        seqs = k_prev_words  # (k, 1)
        # Tensor to store top k sequences' scores; now they're just 0
        top_k_scores = torch.zeros(k, 1).to(device)  # (k, 1)
        # Lists to store completed sequences and scores
        complete_seqs = list()
        complete_seqs_scores = list()
        # Start decoding
        step = 1
        # s is a number less than or equal to k, because sequences are removed from this process once they hit <end>
        self.LM_decoder.set_prefix(gx)

        while True:
            # first version
            # gx = gx_origin.expand(k, 1, imgs.size(-1))
            # imgs = imgs_origin.expand(k, imgs.size(-2), imgs.size(-1))

            x = self.embed(seqs)  # (1, seq, d_model)
            x = self.pos_encoder(x)  # (seq, 1, d_model)
            x, pad_mask, attn_mask = self.prefuse_gx(x, gx = gx, seq_padding_mask = None)
            
            x = self.LM_decoder(
                src = x, 
                incremental_state = incremental_state,
                mask = attn_mask, 
                src_key_padding_mask = pad_mask
                )

            scores = self.classifier(x[:, -1, :])  # (1, vocab_size)
            scores = F.log_softmax(scores, dim=1)
            # Add
            scores = top_k_scores.expand_as(scores) + scores  # (s, vocab_size)
            # For the first step, all k points will have the same scores (since same k previous words, h, c)
            if step == 1:
                top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)  # (s)
            else:
                # Unroll and find top scores, and their unrolled indices
                top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)  # (s)
            # Convert unrolled indices to actual indices of scores
            prev_word_inds = torch.div(top_k_words, self.vocab_size, rounding_mode="floor")  # (s)
            next_word_inds = top_k_words % self.vocab_size  # (s)
            # Add new words to sequences
            seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)  # (s, step+1)
            # Which sequences are incomplete (didn't reach <end>)?
            incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if next_word != end_unit]
            complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))
            # Set aside complete sequences
            if len(complete_inds) > 0:
                complete_seqs.extend(seqs[complete_inds].tolist())
                complete_seqs_scores.extend(top_k_scores[complete_inds])
            k -= len(complete_inds)  # reduce beam length accordingly
            # Proceed with incomplete sequences
            if k == 0:
                break
            seqs = seqs[incomplete_inds]
            # imgs = imgs[prev_word_inds[incomplete_inds]]
            gx = gx[prev_word_inds[incomplete_inds]]
            top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
            k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)
            # Break if things have been going on too long
            # print(k, step)
            # print(seqs)
            if step > max_len:
                break
            step += 1
            
        # print(seqs)
        if len(complete_seqs_scores) != 0:
            i = complete_seqs_scores.index(max(complete_seqs_scores))
            seq = complete_seqs[i]
        else:
            seq = []

        return seq