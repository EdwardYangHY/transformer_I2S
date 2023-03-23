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
from models import PositionalEncoding, TransformerLM, TransformerConditionedLM
from image_encoder import DinoResEncoder, ViTEncoder
from PureT_encoder import Encoder as refine_encoder
from typing import Optional, Any, Union, Callable
from torch import Tensor

import yaml
# with open('../../config.yml', 'r') as yml:
#     config = yaml.safe_load(yml)

# global MAX_LEN, BATCH_FIRST
# MAX_LEN = int(config['data']['max_len']) + 2
# refine_encoder_params = config["i2u"]["refine_encoder_params"]

class custom_TransformerDecoderLayer(nn.TransformerDecoderLayer):
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
        self.threshold = tau
        # self.fc_alpha = nn.Linear(d_model + d_model, d_model)

    def forward(
        self, 
        tgt: Tensor, 
        memory: Tensor, 
        tgt_mask: Optional[Tensor] = None, 
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None, 
        memory_key_padding_mask: Optional[Tensor] = None
        ) -> Tensor:
        x = tgt
        # with out mha module, this becomes a encoder
        # TODO: add gate for mha head
        if self.norm_first:
            # x = x + self._sa_block(self.norm1(x), tgt_mask, tgt_key_padding_mask)
            # x = x + self._mha_block(self.norm2(x), memory, memory_mask, memory_key_padding_mask)
            # x = x + self._ff_block(self.norm3(x))

            # out put of self attention. by pass mha block will make it transformer encoder
            x = x + self._sa_block(self.norm1(x), tgt_mask, tgt_key_padding_mask)

            # residue is original LM information
            # mha_attention is attend to visual information
            residue = x
            mha_attention = self._mha_block(self.norm2(x), memory, memory_mask, memory_key_padding_mask)

            '''without gates, original setting'''
            # enc_atten = residue + mha_attention
            # x_origin = enc_atten + self._ff_block(self.norm3(enc_atten))

            '''
            Implementation in original paper
            In their released code, alpha is cat by residue and mha_attention
            and passed through an additional Linear layer.
            '''
            alpha = torch.sigmoid(residue)
            # alpha = torch.sigmoid(self.fc_alpha(torch.cat([residue, mha_attention], -1)))
            LM_gate = torch.where(alpha > self.threshold, torch.ones_like(alpha), torch.zeros_like(alpha))
            visual_gate = torch.where(alpha < 1 - self.threshold, torch.ones_like(alpha), torch.zeros_like(alpha))
            enc_atten = alpha* LM_gate* residue + (1-alpha)* visual_gate* mha_attention
            x = enc_atten + self._ff_block(self.norm3(enc_atten))

        else:
            raise NotImplementedError
            # x = self.norm1(x + self._sa_block(x, tgt_mask, tgt_key_padding_mask))
            # x = self.norm2(x + self._mha_block(x, memory, memory_mask, memory_key_padding_mask))
            # x = self.norm3(x + self._ff_block(x))
        return x

class TransformerConditionedLM_gated(TransformerConditionedLM):
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
        fine_tune_image_encoder: bool = False,
        use_refine_encoder: bool = False,
        use_global_feature: bool = False,
        AR: bool = True,
        refine_encoder_params: dict = None,
        tau: float = 0.2,
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
            refine_encoder_params,
        )
        # Decoder
        decoder_norm = nn.LayerNorm(d_model, eps=layer_norm_eps)
        # decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=4*d_model, dropout= self.dropout,
        #                                            activation=activation, batch_first=batch_first, norm_first=norm_first)
        decoder_layer = custom_TransformerDecoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=4*d_model, dropout= self.dropout,
                                                    activation=activation, batch_first=batch_first, norm_first=norm_first, tau=tau)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers, decoder_norm)

        # self.classifier = nn.Linear(d_model, vocab_size)
        self.init_weights()

    

