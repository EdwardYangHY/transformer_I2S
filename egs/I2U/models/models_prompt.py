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
from models import PositionalEncoding, TransformerLM, TransformerConditionedLM, TransformerSentenceLM
from image_encoder import DinoResEncoder, ViTEncoder, DinoResEncoder_NoPool, DinoResEncoder_Pool
from PureT_encoder import Encoder as refine_encoder
from typing import Optional, Any, Union, Callable
from torch import Tensor

import yaml

# class TransformerPrefixLM(TransformerConditionedLM):
class TransformerPrefixLM(TransformerSentenceLM):

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
        sentence_embed: int = 8,
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
        if use_global_feature:
             self.prefix_length = refine_encoder_params["input_resolution"]**2 + 1
        else:
             self.prefix_length = refine_encoder_params["input_resolution"]**2
        
        if use_sentence_encoder:
            self.prefix_length += 1
        
        decoder_norm = nn.LayerNorm(d_model, eps=layer_norm_eps)
        decoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=4*d_model, dropout= self.dropout,
                                                   activation=activation, batch_first=batch_first, norm_first=norm_first)
        self.LM_decoder = nn.TransformerEncoder(decoder_layer, num_layers, decoder_norm)
        if image_backbone.upper() == "RESNET":
            self.image_encoder = DinoResEncoder_Pool(encoded_image_size=refine_encoder_params["input_resolution"])
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
        """
            To perform prefix of the input, here we cat [img, gx]
            Say img is a tensor of shape: [B, resolution^2, 2048->1024]
            gx will be of shape:          [B, 1, 1024]
            Concaten gx will be:          [B, resolution^2+1, 1024]
        """
        imgs, gx = self.image_encoder(imgs)
        imgs = self.image_encoder_embedding(imgs) # (Batch, 7*7, d_model)
        gx = imgs.mean(1)
        if self.use_refine_encoder:
            gx, imgs = self.refine_encoder(imgs)
        if gx.dim() == 2:
            gx = gx.unsqueeze(dim = 1)
        # cat [img, gx]
        if self.use_global_feature:
            imgs = torch.cat([imgs, gx], dim = 1)
        return imgs, gx

    def prefuse_gx(self, x, imgs, seq_padding_mask):
        """
            To perform prefix of the input,
            prefusing_gx will be different than previous method:

            Previously, if use_global_feature == True:
                we will cat [gx, x] to : [B, 1+T, 1024]
                and modify padding mask and atten mask:
                For padding, we cat none-padding [B, 1, 1024] to previous mask;
                For atten, we generate VALL-E like mask which always attend to gx.
            Else if use_global_feature == False:
                This section will do nothing but give back previous x and padding_mask,
                with a newly generated attn_mask

            Now, due to prefix, we have img' = [img, gx].
            (We should also consider Sentence Embedding, [img, Sentence])
            Which means we always concat sth to previous x:
            If use_global_feature == True:
                We concate [img', x] to be: [B, resolution**2+1+T, 1024]
                so we have prefix_length = resolution^2+1
            Else if use_global_feature == False:
                We concate [img'[:-1] == img, x] to: [B, resolution**2+T, 1024]
                so we have prefix_length = resolution**2
            So new x will be [B, prefix_length+T, 1024]
            Now for the masks:
                For padding, we cat none-padding [B, prefix_length, 1024] to previous mask;
                For atten, we generate VALL-E like mask which always attend to prefix.
        """

        gx = imgs
        if gx.size(0) == 1 and x.size(0) != 1:
            # expand gx to the batch/beam size of x
            gx = gx.expand(x.size(0), 1, gx.size(2))

        # if self.use_global_feature:
        #     gx = gx
        # else:
        #     gx = gx[:,:-1,:]
        # prefix_length = gx.size(1)
        # if prefix_length <= 1:
        #     raise ValueError

        prefix_length = gx.size(1)
        assert prefix_length == self.prefix_length, f"Prefused prefix length {prefix_length} not equals to cfg {self.prefix_length}"

        decoder_input = torch.cat([gx,x], dim=1)

        if seq_padding_mask is not None:
            decoder_input_padding_mask = self.generate_cat_key_padding_mask(img_len=prefix_length, padding_mask=seq_padding_mask).to(x.device)
        else:
            decoder_input_padding_mask = seq_padding_mask
        decoder_input_attn_mask = self.generate_VALLE_mask(img_len=prefix_length, seq_len=x.size(dim=1)).to(x.device)

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

        if self.use_sentence_encoder:
            z, kl_loss = self.encode_x(x, seq_len, seq_padding_mask, verbose=False)
            # then cat z with imgs ([imgs, gx])
            z = z.unsqueeze(dim=1)
            imgs = torch.cat([imgs, z], dim = 1)

        # Note that, now imgs = [imgs, gx (optional), z (optional)]
        # so img_len
        img_len = imgs.size(1)

        decoder_input, decoder_input_padding_mask, decoder_input_attn_mask = self.prefuse_gx(x, imgs, seq_padding_mask)

        if self.AR:
            decoder_output = self.LM_decoder(
                src = decoder_input, 
                mask = decoder_input_attn_mask, 
                src_key_padding_mask = decoder_input_padding_mask
                )
        else:
            decoder_output = self.LM_decoder(
                src = decoder_input, 
                src_key_padding_mask = decoder_input_padding_mask
                )

        # if self.use_global_feature:
        #     decoder_output = decoder_output[:, img_len+1:, :] # the output lenth should be reduced
        # else:
        #     decoder_output = decoder_output[:, img_len:, :]
        decoder_output = decoder_output[:, img_len:, :]

        decoder_output = self.classifier(decoder_output)
        
        if self.use_sentence_encoder:
            return decoder_output, encoded_seq, decode_lenths, sort_ind, kl_loss
        else:
            return decoder_output, encoded_seq, decode_lenths, sort_ind
    
    def decode(self, image=None, start_unit: int = None, end_unit: int = None, 
               action: torch.Tensor = None, max_len: int = 500, beam_size: int = 5):
        self.eval()
        with torch.no_grad():
            device = next(self.parameters()).device

            if action is not None:
                """Here action to image should be caten [imgs, gx, z]"""
                imgs, gx = self.action_to_image(action)
                # here imgs is [imgs, gx, z] if gx and z are used
            elif image is not None:
                img = image.to(device)
                # print(device)
                assert img.dim() == 4, "Input should be sized: [1, C, H, W]"
                assert img.size(0) == 1, "Inference one image at a time"
                # imgs, gx = self.image_encoder(img)
                imgs, gx = self.get_image_features(img)
                # here imgs is [imgs, gx] if gx is used
            else:
                print("Input at least one from: Image or Action")
                raise ValueError
            
            # ''' Here, we have gx = [img, gx] as in the forward function '''
            # assert gx.size(1) > 1, "Image and gx are not concatenated. Please check."
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
        # self.LM_decoder.set_prefix(gx)
        self.LM_decoder.set_prefix(imgs)

        while True:
            # first version
            # gx = gx_origin.expand(k, 1, imgs.size(-1))
            # imgs = imgs_origin.expand(k, imgs.size(-2), imgs.size(-1))

            x = self.embed(seqs)  # (1, seq, d_model)
            x = self.pos_encoder(x)  # (seq, 1, d_model)
            x, pad_mask, attn_mask = self.prefuse_gx(x, imgs, seq_padding_mask = None)
            
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
            # gx = gx[prev_word_inds[incomplete_inds]]
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



class prefix_TransformerEncoderLayer(nn.TransformerEncoderLayer):
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

        # 可能需要独立在 其之外： 因为需要frozen 整个LM的参数 还需要加载
        # self.key_prompt_embed = nn.Linear(d_model, d_model)
        # self.value_promt_embed = nn.Linear(d_model, d_model)

    def forward(
            self, 
            src: Tensor,
            key_prompt: Tensor,
            value_prompt: Tensor,
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
                attn_mask = src_mask, 
                key_padding_mask = src_key_padding_mask
            ) 
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block_prompt(x, key_prompt, value_prompt, src_mask, src_key_padding_mask))
            x = self.norm2(x + self._ff_block(x))
        return x

    def _sa_block_prompt(
            self,
            x: Tensor,
            k_prompt: Tensor,
            v_prompt: Tensor,
            attn_mask: Optional[Tensor], 
            key_padding_mask: Optional[Tensor]
        ) -> Tensor:
        """ 
        Customized Self-attention Block for prefix prompt tuning.
        Ref: Prefix-tuning https://arxiv.org/abs/2101.00190

        originally, (..., need_weights=Flase)[0] outputs no atten.
        To
        """
        # x_1, atten_1 = self.self_attn(query=x[:,-1:,:], key=k_prompt[:,-1:,:], value=v_prompt[:,-1:,:],
        #                    attn_mask=attn_mask[-1:, -1:],
        #                    key_padding_mask=key_padding_mask,
        #                    need_weights=True)# [0]

        x, atten = self.self_attn(query=x, key=k_prompt, value=v_prompt,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=True)# [0]
        

        
        return self.dropout1(x)

class prefix_TransformerEncoder(nn.TransformerEncoder):
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
            norm, 
            enable_nested_tensor, 
            mask_check
        )
        self.d_model = self.layers[0].d_model
        self.input_prompt_embedder = nn.Conv1d(in_channels=self.d_model, out_channels=self.d_model, kernel_size=1)
        self.key_embedders = nn.ModuleList([nn.Conv1d(in_channels=self.d_model, out_channels=self.d_model, kernel_size=1) for i in range(num_layers)])
        self.value_embedders = nn.ModuleList([nn.Conv1d(in_channels=self.d_model, out_channels=self.d_model, kernel_size=1) for i in range(num_layers)])
        # # --------------------------------------------------------------------------------------------------------
        # self.key_embedders = nn.ModuleList([nn.Linear(self.d_model, self.d_model) for i in range(num_layers)])
        # self.value_embedders = nn.ModuleList([nn.Linear(self.d_model, self.d_model) for i in range(num_layers)])
        # # --------------------------------------------------------------------------------------------------------
        self.prompt_activation = nn.ReLU()
        self.prompt_dropout = nn.Dropout(p=0.05)
        self.prefix_length = prefix_length
        self.prefix_cache = None

        first_layer = self.layers[0]
        # assert isinstance(first_layer, prefix_TransformerEncoderLayer), f"{first_layer} is not a prefix_TransformerEncoderLayer"
    
    def set_prefix(self,src: Tensor):
        self.prefix_cache = {}

        input_prefix = src.clone()
        input_prefix = input_prefix[:,:self.prefix_length,:]
        self.prefix_cache["prefix"] = input_prefix
        input_prefix_prompt = self.input_prompt_embedder(input_prefix.permute(0,2,1)).permute(0,2,1)
        input_prefix_prompt = self.prompt_dropout(self.prompt_activation(input_prefix_prompt))
        self.prefix_cache["input_prompt"] = input_prefix_prompt

        self.prefix_cache["key_prompt"] = []
        self.prefix_cache["value_prompt"] = []

        for key_embedder, value_embedder in zip(self.key_embedders, self.value_embedders):
            key_prefix_prompt = key_embedder(input_prefix.permute(0,2,1)).permute(0,2,1)
            key_prefix_prompt = self.prompt_dropout(self.prompt_activation(key_prefix_prompt))
            self.prefix_cache["key_prompt"].append(key_prefix_prompt)

            value_prefix_prompt = value_embedder(input_prefix.permute(0,2,1)).permute(0,2,1)
            value_prefix_prompt = self.prompt_dropout(self.prompt_activation(value_prefix_prompt))
            self.prefix_cache["value_prompt"].append(value_prefix_prompt)

    def forward(
            self, 
            src: Tensor, 
            # prefix_embedders: None,
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

            if not self.training:
                output = mod(
                    output, 
                    key_prompt=key_prompt,
                    value_prompt=value_prompt,
                    src_mask=mask, 
                    src_key_padding_mask=src_key_padding_mask
                    )
            else:
                output = mod(
                    output, 
                    key_prompt=key_prompt,
                    value_prompt=value_prompt,
                    src_mask=mask, 
                    src_key_padding_mask=src_key_padding_mask
                    )
            
        # for mod, key_embedder, value_embedder in zip(self.layers, self.key_embedders, self.value_embedders):
        #     """
        #         Here we assert BatchFirst is True
        #         Embedder is a conv1d layer, we should permute [B, T, dim] -> [B, dim, T]
        #     """ 

        #     """
        #         计算大量冗余：
        #         Prefix是不会变的 所以不需要每次都计算
        #     """
        #     key_prompt = output.clone()
        #     key_prefix_prompt = key_embedder(input_prefix.permute(0,2,1)).permute(0,2,1)
        #     key_prefix_prompt = self.prompt_dropout(self.prompt_activation(key_prefix_prompt))
        #     key_prompt[:,:self.prefix_length,:] = key_prefix_prompt
            
        #     value_prompt = output.clone()
        #     value_prefix_prompt = value_embedder(input_prefix.permute(0,2,1)).permute(0,2,1)
        #     value_prefix_prompt = self.prompt_dropout(self.prompt_activation(value_prefix_prompt))
        #     value_prompt[:,:self.prefix_length,:] = value_prefix_prompt
            

        #     output = mod(
        #         output, 
        #         key_prompt=key_prompt,
        #         value_prompt=value_prompt,
        #         src_mask=mask, 
        #         src_key_padding_mask=src_key_padding_mask
        #         )

        if self.norm is not None:
            output = self.norm(output)

        return output

    def freeze(self):
        """
            Only Fix tranformer layers, not prompt Embeddings.
        """
        for p in self.layers.parameters():
            p.requires_grad = False

    def load_PuLM(self, model_dict):
        """
            Load pretrained uLM's params to each layer
        """
        pass        

class prefix_Transformer(TransformerPrefixLM):
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
            use_sentence_encoder,
            sentence_embed,
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
        decoder_layer = prefix_TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=4*d_model, dropout= self.dropout,
                                                   activation=activation, batch_first=batch_first, norm_first=norm_first)
        self.LM_decoder = prefix_TransformerEncoder(decoder_layer, num_layers, self.prefix_length, decoder_norm)


    def freeze_LM(self):
        return self.LM_decoder.freeze()
    
    def load_Pretrained_LM(self, LM_path):
        return super().load_Pretrained_LM(LM_path)

    def action_to_image(self, action, beam_size):
        if action.size(-1) > 64*2048:
            fmap = action[:, :64*2048].view(1, 64, 2048)
            gx = fmap.mean(1)
            if self.use_refine_encoder:
                gx, fmap = self.refine_encoder(fmap)
            embed = action[:, 64*2048:]
            embed = self.make_memory(embed)
            embed = embed.unsqueeze(1)
            if self.use_global_feature:
                m = torch.cat([fmap, gx, embed], dim=1)  # (1, 50, d_model)
            else:
                m = torch.cat([fmap, embed], dim=1)  # (1, 50, d_model)
            m = m.expand(beam_size, m.size(1), 2048)
        else:
            fmap = action[:, :64*2048].view(1, 64, 2048)
            gx = fmap.mean(1)
            if self.use_refine_encoder:
                gx, fmap = self.refine_encoder(fmap)
            if self.use_global_feature:
                m = torch.cat([fmap, gx], dim=1)  # (1, 50, d_model)
            m = fmap.expand(beam_size, m.size(1), 2048)
        return m, gx