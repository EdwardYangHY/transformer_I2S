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

# sys.path.append("/net/papilio/storage2/yhaoyuan/transformer_I2S/egs/I2U/models")
from models import PositionalEncoding, TransformerLM, TransformerConditionedLM, TransformerSentenceLM
from models_custom import custom_TransformerDecoderLayer, TransformerConditionedLM_gated
from image_encoder import DinoResEncoder, ViTEncoder, DinoResEncoder_NoPool, DinoResEncoder_Pool
from PureT_encoder import Encoder as refine_encoder
from typing import Optional, Any, Union, Callable
from torch import Tensor
from load_pretrained_uLM import uLM2decoder

import yaml
# with open('../../config.yml', 'r') as yml:
#     config = yaml.safe_load(yml)

# global MAX_LEN, BATCH_FIRST
# MAX_LEN = int(config['data']['max_len']) + 2
# refine_encoder_params = config["i2u"]["refine_encoder_params"]

class TransformerConditionedLM_FixedImg(TransformerConditionedLM):
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
        if image_backbone.upper() == "RESNET":
            self.image_encoder = DinoResEncoder_NoPool()
            self.image_encoder.fine_tune(fine_tune_image_encoder)
        else:
            raise NotImplementedError
        self.image_encoder_embedding = nn.Linear(2048, d_model)
    
    def get_image_features(self, imgs):
        imgs, gx = self.image_encoder(imgs)       # (Batch, 7*7, 2048)
        imgs = self.image_encoder_embedding(imgs) # (Batch, 7*7, d_model)
        gx = imgs.mean(1)
        if self.use_refine_encoder:
            gx, imgs = self.refine_encoder(imgs)
        return imgs, gx
        # The image encoder is only DinoResNet.
        # The out put is (Batch, (input_resolution/32)^2, 2048)
    
    def action_to_image(self, action):
        assert action.size(-1) >= 49*2048, "Action size too small"
        imgs = action[:, :49*2048].view(1, 49, 2048) # (1, 7*7, 2048)
        imgs = self.image_encoder_embedding(imgs)    # (1, 7*7, d_model)
        gx = imgs.mean(1)
        if self.use_refine_encoder:
            gx, imgs = self.refine_encoder(imgs)
        return imgs, gx


class TransformerSentenceLM_FixedImg_Pool(TransformerSentenceLM):
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
        use_sentence_encoder: bool = True,
        sentence_embed: int = 16,
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
            use_sentence_encoder,
            sentence_embed,
            fine_tune_image_encoder,
            use_refine_encoder,
            use_global_feature,
            AR,
            refine_encoder_params
        )
        # 加到基类: TransformerLM 里了
        # self.nhead = nhead
        # self.num_layers = num_layers
        if image_backbone.upper() == "RESNET":
            # self.image_encoder = DinoResEncoder_NoPool()
            self.image_encoder = DinoResEncoder_Pool(encoded_image_size = refine_encoder_params["input_resolution"]) # out = [batch, 2048, resolution/32, resolution/32]
            self.image_encoder.fine_tune(fine_tune_image_encoder)
        else:
            raise NotImplementedError
        
        # encoded_image_size = refine_encoder_params["input_resolution"]
        # self.image_pooling = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))
        self.image_encoder_embedding = nn.Linear(2048, d_model)
    
    def get_image_features(self, imgs):
        imgs, gx = self.image_encoder(imgs)       # (Batch, 7*7, 2048)
        imgs = self.image_encoder_embedding(imgs) # (Batch, 7*7, d_model)
        gx = imgs.mean(1)
        if self.use_refine_encoder:
            gx, imgs = self.refine_encoder(imgs)
        return imgs, gx
        # The image encoder is only DinoResNet.
        # The out put is (Batch, encoded_image_size^2, 2048)
    
    def action_to_image(self, action, beam_size):
        """
            Why 49*2048?
            This is decided by the feature extractor of RL.
            The image resolution in previous task is 224*224.
            And ResNet-Dino extract feature map by (resolution/32)^2,
            and we ignored the classification layers to get raw image
            features.
            So the features of img originally from ResNet is:
            [Batch, 224/32, 224/32, 2048]
        """
        if action.size(-1) > 49*2048:
            fmap = action[:, :49*2048].view(1, 49, 2048) # (1, 49, 2048)
            fmap = self.image_encoder_embedding(fmap) # (1, 49, d_model)
            gx = fmap.mean(1)
            if self.use_refine_encoder:
                gx, fmap = self.refine_encoder(fmap)
            embed = action[:, 49*2048:]
            embed = self.make_memory(embed)
            embed = embed.unsqueeze(1)
            m = torch.cat([fmap, embed], dim=1)  # (1, 50, d_model)
            m = m.expand(beam_size, 50, self.d_model)
        else:
            fmap = action[:, :49*2048].view(1, 49, 2048)
            fmap = self.image_encoder_embedding(fmap) # (1, 49, d_model)
            gx = fmap.mean(1)
            if self.use_refine_encoder:
                gx, fmap = self.refine_encoder(fmap)
            m = fmap.expand(beam_size, 49, self.d_model)
        return m, gx
    
    def load_Pretrained_LM(self, LM_path):
        """
            Load pretrained uLM's weights to current decoder.
            We ignored:
                1. Positional Encoding. This will not change.
                2. Multi-head attention + norm layer. (no mha in LM)
            
            So how to set mha layer is up to the need, since mha is
            initialized randomly.

            uLM2decoder takes (current state dict, uLM state dict as 
            input), will map the correspond value in uLM to current 
            state dict, and return mapped state dict.

            NOTE that:
            uLM is 1024 dim, 16 heads, 12 layers.
            decoder has to be the same structure.
        """
        assert self.d_model == 1024, f"Expect d_model: 1024, get d_model: {self.d_model}"
        assert self.nhead == 16, f"Expect nhead: 16, get nhead: {self.nhead}"
        assert self.num_layers == 12, f"Expect layers: 12, get layers: {self.num_layers}"

        print(f"Load uLM weights from path: {LM_path}")
        LM_model = torch.load(LM_path)
        LM_state_dict = LM_model["model_state_dict"]
        current_state_dict = self.state_dict()
        loaded = uLM2decoder(current_state_dict, LM_state_dict)
        self.load_state_dict(loaded)
    
    def freeze_LM(self):
        """
            Only Fix Decoder layers, not Embedding or Classifier
        """
        for p in self.decoder.parameters():
            p.requires_grad = False
    
    def freeze_LM_hard(self):
        """
            Fix Decoder layers, Embedding and Classifier
        """
        for p in self.embed.parameters():
            p.requires_grad = False
                
        for p in self.decoder.parameters():
            p.requires_grad = False
        
        for p in self.classifier.parameters():
            p.requires_grad = False

class TransformerSentenceLM_FixedImg_gated(TransformerSentenceLM_FixedImg_Pool):
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
        use_sentence_encoder: bool = True,
        sentence_embed: int = 16,
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
            use_sentence_encoder,
            sentence_embed,
            fine_tune_image_encoder,
            use_refine_encoder,
            use_global_feature,
            AR,
            refine_encoder_params
        )
        decoder_norm = nn.LayerNorm(d_model, eps=layer_norm_eps)
        # decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=4*d_model, dropout= self.dropout,
        #                                            activation=activation, batch_first=batch_first, norm_first=norm_first)
        decoder_layer = custom_TransformerDecoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=4*d_model, dropout= self.dropout,
                                                    activation=activation, batch_first=batch_first, norm_first=norm_first, tau=tau)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers, decoder_norm)

        # self.classifier = nn.Linear(d_model, vocab_size)
        self.init_weights()
    
    # def load_Pretrained_LM(self, LM_path):
    #     print(f"Load uLM weights from path: {LM_path}")
    #     LM_model = torch.load(LM_path)
    #     LM_state_dict = LM_model["model_state_dict"]
    #     current_state_dict = self.state_dict()
    #     loaded = uLM2decoder(current_state_dict, LM_state_dict)
    #     self.load_state_dict(loaded)
