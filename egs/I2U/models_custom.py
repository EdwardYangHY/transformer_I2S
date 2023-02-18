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
from image_encoder import DinoResEncoder, DinoResEncoder_FixPooling, ViTEncoder
from PureT_encoder import Encoder as refine_encoder
from typing import Optional, Any, Union, Callable
from torch import Tensor

import yaml
with open('../../config.yml', 'r') as yml:
    config = yaml.safe_load(yml)

global MAX_LEN, BATCH_FIRST
MAX_LEN = int(config['data']['max_len']) + 2
refine_encoder_params = config["i2u"]["refine_encoder_params"]

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
        device=None, 
        dtype=None
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
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), tgt_mask, tgt_key_padding_mask)
            x = x + self._mha_block(self.norm2(x), memory, memory_mask, memory_key_padding_mask)
            x = x + self._ff_block(self.norm3(x))
        else:
            x = self.norm1(x + self._sa_block(x, tgt_mask, tgt_key_padding_mask))
            x = self.norm2(x + self._mha_block(x, memory, memory_mask, memory_key_padding_mask))
            x = self.norm3(x + self._ff_block(x))
        return x

class TransformerConditionedLM_custom(TransformerLM):
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
        AR: bool = True
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
        )
        self.fine_tune_image_encoder = fine_tune_image_encoder
        self.dropout = dropout
        self.AR = AR
        self.use_refine_encoder = use_refine_encoder
        self.use_global_feature = use_global_feature

        self.LM_decoder = None

        # Get Image backbone
        if image_backbone.upper() == "RESNET":
            self.image_encoder = DinoResEncoder(encoded_image_size=refine_encoder_params["input_resolution"] , embed_dim=d_model)  # Image feature is [Batch, 14*14, d_model]
            # self.image_encoder = DinoResEncoder_NoPooling(embed_dim=d_model) 
            self.image_encoder.fine_tune(self.fine_tune_image_encoder)
        elif image_backbone.upper() == "VIT":
            self.image_encoder = ViTEncoder(embed_dim=d_model) 
        elif image_backbone.upper() == "ST":
            print("Unimplemented Yet")
            return None
        else:
            print("Please choose an image backbone: ResNet, ViT, SW")
            return None
        
        if self.use_refine_encoder:
            resolution = refine_encoder_params["input_resolution"]
            assert type(resolution) == int, "Input Resolution for refine encoder has to be an int."
            self.refine_encoder = refine_encoder(
                embed_dim=d_model, 
                input_resolution=(resolution, resolution), 
                depth=int(refine_encoder_params["depth"]), 
                num_heads=int(refine_encoder_params["num_heads"]), 
                window_size=int(refine_encoder_params["window_size"]),  # =14 退化为普通MSA结构
                shift_size=int(refine_encoder_params["shift_size"]),    # =0  无SW-MSA，仅W-MSA
                mlp_ratio=int(refine_encoder_params["mlp_ratio"]),
                dropout=self.dropout,
                use_gx=self.use_global_feature
            )

        # self.embed = nn.Embedding(vocab_size, d_model)
        # self.pos_encoder = PositionalEncoding(d_model)

        # Pre-fusion
        if self.use_global_feature:
            # Refering to PureT
            self.prefusion_layer = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.ReLU(),
                nn.Dropout(self.dropout)
            )
            self.prefusion_norm = nn.LayerNorm(d_model, eps=layer_norm_eps)
        
        # Decoder
        decoder_norm = nn.LayerNorm(d_model, eps=layer_norm_eps)
        # decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=4*d_model, dropout= self.dropout,
        #                                            activation=activation, batch_first=batch_first, norm_first=norm_first)
        decoder_layer = custom_TransformerDecoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=4*d_model, dropout= self.dropout,
                                                    activation=activation, batch_first=batch_first, norm_first=norm_first)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers, decoder_norm)

        # self.classifier = nn.Linear(d_model, vocab_size)
        self.init_weights()
    
    def prefuse_gx(self, x, gx, seq_padding_mask):
        if self.use_global_feature:
            if gx.dim() == 2:
                # Here we don't use the global feature the same way as PureT:
                # We cat [gx, x] in dim = 1, instead of dim = 2 (expanding the feature)
                gx = gx.unsqueeze(dim = 1)
            decoder_input = torch.cat([gx,x], dim = 1)
            short_cut = decoder_input
            decoder_input = self.prefusion_layer(decoder_input)
            decoder_input = decoder_input + short_cut
            decoder_input = self.prefusion_norm(decoder_input)
            # besides, the padding mask should be changed
            decoder_input_padding_mask = self.generate_cat_key_padding_mask(img_len=1, padding_mask=seq_padding_mask).to(x.device)
            decoder_input_attn_mask = self.generate_VALLE_mask(img_len=1, seq_len=decoder_input.size(dim=1)-1).to(x.device)
        else:
            decoder_input = x
            decoder_input_padding_mask = seq_padding_mask
            decoder_input_attn_mask = self.generate_square_subsequent_mask(decoder_input.size(dim=1)).to(x.device)
        return decoder_input, decoder_input_padding_mask, decoder_input_attn_mask

    def forward(self, imgs, seq, seq_padding_mask, seq_len):

        seq_len, sort_ind = seq_len.sort(dim=0, descending=True)
        imgs = imgs[sort_ind]
        encoded_seq = seq[sort_ind]
        decode_lenths = (seq_len - 1).tolist()
        max_length = max(decode_lenths)
        seq = encoded_seq[:, :max_length]
        seq_padding_mask = seq_padding_mask[:, :max_length]

        # get seq into embeddings
        x = self.embed(seq)
        x = self.pos_encoder(x)
        '''
            If you use ResNet:
            gx is the mean pooling of the image features.
            If you use ViT:
            gx is the classifier token.
        '''
        imgs, gx = self.image_encoder(imgs)
        if self.use_refine_encoder:
            '''
                Here if use_global_feature is True:
                gx will be extracted in advance and refined together.
                Otherwise,
                gx will be the mean pooling of the refined features.
            '''
            gx, imgs = self.refine_encoder(imgs)
            
        decoder_input, decoder_input_padding_mask, decoder_input_attn_mask = self.prefuse_gx(x, gx, seq_padding_mask)

        if self.AR:
            decoder_output = self.decoder(
                tgt = decoder_input, 
                memory = imgs, 
                tgt_mask = decoder_input_attn_mask, 
                tgt_key_padding_mask = decoder_input_padding_mask
                )
        else:
            # the only difference between AR and NAR Transformer is:
            # whether the decoder can feel all decoder_input, indifferent to decode steps
            decoder_output = self.decoder(
                tgt = decoder_input, 
                memory = imgs,
                tgt_key_padding_mask = decoder_input_padding_mask
                )

        if self.use_global_feature:
            decoder_output = decoder_output[:, 1:, :] # the output lenth should be reduced
        decoder_output = self.classifier(decoder_output)
        return decoder_output, encoded_seq, decode_lenths, sort_ind
    
    def decode(
        self,
        img,
        start_unit: int,
        end_unit: int,
        max_len: int = 500,
        beam_size: int = 5,
        ):
        self.eval()
        with torch.no_grad():
            device = next(self.parameters()).device
            img = img.to(device)
            # print(device)

            assert img.dim() == 4, "Input should be sized: [1, C, H, W]"
            assert img.size(0) == 1, "Inference one image at a time"

            imgs, gx = self.image_encoder(img)
            if self.use_refine_encoder:
                '''
                here gx [1, d_model]
                imgs    [1, 196, d_model]
                '''
                gx, imgs = self.refine_encoder(imgs)
            gx_origin = gx
            imgs_origin = imgs
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

            while True:
                gx = gx_origin.expand(k, 1, imgs.size(-1))
                imgs = imgs_origin.expand(k, imgs.size(-2), imgs.size(-1))
                x = self.embed(seqs)  # (1, seq, d_model)
                x = self.pos_encoder(x)  # (seq, 1, d_model)
                if self.use_global_feature:
                    decoder_input = torch.cat([gx,x], dim = 1)
                    short_cut = decoder_input
                    decoder_input = self.prefusion_layer(decoder_input)
                    decoder_input = decoder_input + short_cut
                    decoder_input = self.prefusion_norm(decoder_input)
                else:
                    decoder_input = x
                mask = self.generate_VALLE_mask(1, x.size(1)).to(device)
                # torch.save(decoder_input, "./inter_tensor.pt")
                # if step == 90:
                #     decoder_input = torch.load("./inter_tensor.pt")

                # TODO:
                '''
                    def beam_search():
                '''
                x = self.decoder(decoder_input, imgs, mask) # 解码时必须有mask， 为什么？
                # x = self.decoder(decoder_input, imgs)
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
                imgs = imgs[prev_word_inds[incomplete_inds]]
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
