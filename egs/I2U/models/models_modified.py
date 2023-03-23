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
from image_encoder import DinoResEncoder, ViTEncoder, DinoResEncoder_NoPool, DinoResEncoder_Raw
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

class TransformerSentenceLM_FixedImg(TransformerSentenceLM):
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
            self.image_encoder = DinoResEncoder_Raw() # out = [batch, 2048, resolution/32, resolution/32]
            self.image_encoder.fine_tune(fine_tune_image_encoder)
        else:
            raise NotImplementedError
        
        encoded_image_size = refine_encoder_params["input_resolution"]
        self.image_pooling = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))
        self.image_encoder_embedding = nn.Linear(2048, d_model)
    
    def get_image_features_(self, imgs):
        imgs, gx = self.image_encoder(imgs)       # (Batch, 7*7, 2048)
        imgs = self.image_encoder_embedding(imgs) # (Batch, 7*7, d_model)
        gx = imgs.mean(1)
        if self.use_refine_encoder:
            gx, imgs = self.refine_encoder(imgs)
        return imgs, gx
        # The image encoder is only DinoResNet.
        # The out put is (Batch, (input_resolution/32)^2, 2048)
    
    def get_image_features(self, imgs):
        # input image size is (Batch, 2048, reso/32, reso/32)
        imgs = self.image_encoder(imgs)           # (Batch, 2048, reso/32, reso/32)
        imgs = self.image_pooling(imgs)           # (Batch, 2048, encoded, encoded)
        imgs = imgs.permute(0,2,3,1)              # (Batch, encoded, encoded, 2048)
        imgs = imgs.view(imgs.size(0), -1, imgs.size(-1)) # (Batch, encoded^2, 2048)
        imgs = self.image_encoder_embedding(imgs) # (Batch, 7*7, d_model)
        gx = imgs.mean(1)
        if self.use_refine_encoder:
            gx, imgs = self.refine_encoder(imgs)
        return imgs, gx
        # The image encoder is only DinoResNet.
        # The out put is (Batch, (input_resolution/32)^2, 2048)
    
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


class TransformerSentenceLM_FixedImg_gated(TransformerSentenceLM_FixedImg):
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


class TransformerPrefixLM(TransformerConditionedLM):

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
        decoder_norm = nn.LayerNorm(d_model, eps=layer_norm_eps)
        decoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=4*d_model, dropout= self.dropout,
                                                   activation=activation, batch_first=batch_first, norm_first=norm_first)
        self.LM_decoder = nn.TransformerEncoder(decoder_layer, num_layers, decoder_norm)
        if image_backbone.upper() == "RESNET":
            self.image_encoder = DinoResEncoder_NoPool()
            self.image_encoder.fine_tune(fine_tune_image_encoder)
        else:
            raise NotImplementedError
        
        self.image_encoder_embedding = nn.Linear(2048, d_model)
    
    def load_Pretrained_LM(self, LM_path):
        """
            This should be easier because the structure is the same
        """
        print(f"Load uLM weights from path: {LM_path}")
        LM_model = torch.load(LM_path)
        LM_state_dict = LM_model["model_state_dict"]
        self.load_state_dict(LM_state_dict, strict=False)
        # raise NotImplementedError
    
    def get_image_features(self, imgs):
        imgs, gx = self.image_encoder(imgs)
        imgs = self.image_encoder_embedding(imgs) # (Batch, 7*7, d_model)
        gx = imgs.mean(1)
        if self.use_refine_encoder:
            gx, imgs = self.refine_encoder(imgs)
        if gx.dim() == 2:
            gx = gx.unsqueeze(dim = 1)
        # cat [img, gx]
        gx = torch.cat([imgs, gx], dim = 1)
        return imgs, gx

    def prefuse_gx(self, x, gx, seq_padding_mask):
        if gx.size(0) == 1 and x.size(0) != 1:
            # expand gx to the batch/beam size of x
            gx = gx.expand(x.size(0), 1, gx.size(2))

        if self.use_global_feature:
            gx = gx
        else:
            gx = gx[:,:-1,:]
        
        img_len = gx.size(1)
        if img_len <= 1:
            raise ValueError

        decoder_input = torch.cat([gx,x], dim=1)

        if seq_padding_mask is not None:
            decoder_input_padding_mask = self.generate_cat_key_padding_mask(img_len=img_len, padding_mask=seq_padding_mask).to(x.device)
        else:
            decoder_input_padding_mask = seq_padding_mask
        decoder_input_attn_mask = self.generate_VALLE_mask(img_len=img_len, seq_len=x.size(dim=1)).to(x.device)

        return decoder_input, decoder_input_padding_mask, decoder_input_attn_mask

    def freeze_LM(self):
        """
            Only Fix Decoder layers, not Embedding or Classifier
        """
        for p in self.LM_decoder.parameters():
            p.requires_grad = False
    
    def freeze_LM_hard(self):
        """
            Fix Decoder layers, Embedding and Classifier
        """
        for p in self.embed.parameters():
            p.requires_grad = False
                
        for p in self.LM_decoder.parameters():
            p.requires_grad = False
        
        for p in self.classifier.parameters():
            p.requires_grad = False

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

        # function get_image_features, already cat [imgs, fx]
        imgs, gx = self.get_image_features(imgs)
        img_len = imgs.size(1)

        decoder_input, decoder_input_padding_mask, decoder_input_attn_mask = self.prefuse_gx(x, gx, seq_padding_mask)

        if self.AR:
            decoder_output = self.LM_decoder(
                src = decoder_input, 
                # memory = imgs, 
                mask = decoder_input_attn_mask, 
                src_key_padding_mask = decoder_input_padding_mask
                )
        else:
            decoder_output = self.LM_decoder(
                src = decoder_input, 
                # memory = imgs,
                src_key_padding_mask = decoder_input_padding_mask
                )

        if self.use_global_feature:
            decoder_output = decoder_output[:, img_len+1:, :] # the output lenth should be reduced
        else:
            decoder_output = decoder_output[:, img_len:, :]

        decoder_output = self.classifier(decoder_output)
        return decoder_output, encoded_seq, decode_lenths, sort_ind
    
    def decode(self, image=None, start_unit: int = None, end_unit: int = None, 
               action: torch.Tensor = None, max_len: int = 500, beam_size: int = 5):
        self.eval()
        with torch.no_grad():
            device = next(self.parameters()).device

            if action is not None:
                imgs, gx = self.action_to_image(action)
                if gx.dim() == 2:
                    gx = gx.unsqueeze(dim = 1)
                gx = torch.cat([imgs, gx], dim = 1)
            elif image is not None:
                img = image.to(device)
                # print(device)
                assert img.dim() == 4, "Input should be sized: [1, C, H, W]"
                assert img.size(0) == 1, "Inference one image at a time"
                # imgs, gx = self.image_encoder(img)
                imgs, gx = self.get_image_features(img)
            else:
                print("Input at least one from: Image or Action")
                raise ValueError
            
            ''' Here, we have gx = [img, gx] as in the forward function '''
            assert gx.size(1) > 1, "Image and gx are not concatenated. Please check."
            # gx_origin = gx
            # imgs_origin = imgs
            imgs = imgs.expand(beam_size, imgs.size(1), imgs.size(2))
            gx = gx.expand(beam_size, gx.size(1), gx.size(2))
            

            # beam search takes:
            # beam_size， start_unit, end_unit, max_len
            # imgs (decoder memory), [beam, 49, d_model]
            # gx (global feature), [1, (1), d_model] or None.

            seq = self.beam_search(
                imgs=imgs,
                gx=gx,
                beam_size=beam_size,
                start_unit=start_unit,
                end_unit=end_unit,
                max_len=max_len,
                device=device
            )

            return seq

    def beam_search(self, imgs, gx, beam_size, start_unit, end_unit, max_len, device):
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
            # first version
            # gx = gx_origin.expand(k, 1, imgs.size(-1))
            # imgs = imgs_origin.expand(k, imgs.size(-2), imgs.size(-1))

            x = self.embed(seqs)  # (1, seq, d_model)
            x = self.pos_encoder(x)  # (seq, 1, d_model)
            x, pad_mask, attn_mask = self.prefuse_gx(x, gx = gx, seq_padding_mask = None)
            
            x = self.LM_decoder(
                src = x, 
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
