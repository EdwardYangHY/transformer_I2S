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


class PositionalEncoding(nn.Module):
    """
    https://github.com/pytorch/tutorials/blob/master/beginner_source/transformer_tutorial.py
    """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 100+2):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class TransformerAE(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_embed: int = 8,
        d_model: int = 768,
        nhead: int = 8,
        num_layers: int = 6,
        activation="gelu",
        layer_norm_eps: float = 1e-5,
        batch_first: bool = True,
        norm_first: bool = True,
        max_len: int = 100,
        ):
        super().__init__()
        self.d_embed = d_embed

        # Enbedding
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_len+2)
        
        encoder_norm = nn.LayerNorm(d_model, eps=layer_norm_eps)
        decoder_norm = nn.LayerNorm(d_model, eps=layer_norm_eps)

        # Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=4*d_model,
                                                   activation=activation, batch_first=batch_first, norm_first=norm_first)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers, encoder_norm)

        # Decoder
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=4*d_model,
                                                   activation=activation, batch_first=batch_first, norm_first=norm_first)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers, decoder_norm)

        self.mu = nn.Linear(d_model, d_embed)
        self.make_memory = nn.Linear(d_embed, d_model)
        self.classifier = nn.Linear(d_model, vocab_size)
    
    def forward(self, x: torch.Tensor, padding_mask: torch.BoolTensor, seq_len):
        """
        padding_mask:
            True is pad
            False is other
        """
        x = self.embed(x)  # (batch, seq, d_model)
        x = x.permute(1, 0, 2)  # (seq, batch, d_model)
        x = self.pos_encoder(x)  # (seq, batch, d_model)
        x = x.permute(1, 0, 2)  # (batch, seq, d_model)
        z = self.encoder(x, src_key_padding_mask=padding_mask)  # (batch, seq, d_model)
        z = z * padding_mask.logical_not().unsqueeze(2)  # (batch, seq, d_model)
        z = z.sum(dim=1) / seq_len.unsqueeze(1)  # (batch, d_model)
        z = self.mu(z)  # (batch, d_embed)
        print(z)
        z = self.make_memory(z)  # (batch, d_model)
        z = z.unsqueeze(1)  # (batch, 1, d_model)

        x = self.decoder(
            x,
            z,
            tgt_mask=self.generate_square_subsequent_mask(x.size(dim=1)).to(x.device),
            tgt_key_padding_mask=padding_mask,
            )  # (batch, seq, d_model)
        x = self.classifier(x)  # (batch, seq, vocab_size)
        return x, 0
    
    def encode(self, x: torch.Tensor, padding_mask: torch.BoolTensor, seq_len):
        """
        padding_mask:
            True is pad
            False is other
        """
        x = self.embed(x)  # (batch, seq, d_model)
        x = x.permute(1, 0, 2)  # (seq, batch, d_model)
        x = self.pos_encoder(x)  # (seq, batch, d_model)
        x = x.permute(1, 0, 2)  # (batch, seq, d_model)
        z = self.encoder(x, src_key_padding_mask=padding_mask)  # (batch, seq, d_model)
        z = z * padding_mask.logical_not().unsqueeze(2)  # (batch, seq, d_model)
        z = z.sum(dim=1) / seq_len.unsqueeze(1)  # (batch, d_model)
        z = self.mu(z)  # (batch, d_embed)
        z = self.make_memory(z)  # (batch, d_model)
        return z

    def decode(self, z: torch.Tensor, start_unit: int, end_unit: int, max_len: int = 100):
        self.eval()
        with torch.no_grad():
            z = self.make_memory(z)  # (batch, d_model)
            z = z.unsqueeze(1)  # (batch, 1, d_model)

            units = torch.LongTensor([[start_unit]]).to(z.device)

            for _ in range(max_len):
                x = self.embed(units)  # (1, seq, d_model)
                x = x.permute(1, 0, 2)  # (seq, 1, d_model)
                x = self.pos_encoder(x)  # (seq, 1, d_model)
                x = x.permute(1, 0, 2)  # (1, seq, d_model)
                x = self.decoder(
                    x,
                    z,
                    tgt_mask=self.generate_square_subsequent_mask(x.size(dim=1)).to(x.device),
                    )  # (1, seq, d_model)
                x = self.classifier(x[:, -1, :])  # (1, vocab_size)
                
                unit = torch.argmax(x, dim=1, keepdim=True)
                units = torch.cat((units, unit), dim=1)
                
                if end_unit == unit.item():
                    break
        return units[0].tolist()
    
    @staticmethod
    def generate_square_subsequent_mask(sz: int) -> torch.Tensor:
        r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0).
            from https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/transformer.py
        """
        return torch.triu(torch.full((sz, sz), float('-inf')), diagonal=1)


class TransformerVAE(TransformerAE):
    def __init__(
        self,
        vocab_size: int,
        d_embed: int = 16,
        d_model: int = 768,
        nhead: int = 8,
        num_layers: int = 6,
        activation="gelu",
        layer_norm_eps: float = 1e-5,
        batch_first: bool = True,
        norm_first: bool = True,
        ):
        super().__init__(
            vocab_size,
            d_embed,
            d_model,
            nhead,
            num_layers,
            activation,
            layer_norm_eps,
            batch_first,
            norm_first,
        )
        self.log_std = nn.Linear(d_model, d_embed)
    
    def forward(self, x: torch.Tensor, padding_mask: torch.BoolTensor, seq_len):
        """
        padding_mask:
            True is pad
            False is other
        """
        x = self.embed(x)  # (batch, seq, d_model)
        x = x.permute(1, 0, 2)  # (seq, batch, d_model)
        x = self.pos_encoder(x)  # (seq, batch, d_model)
        x = x.permute(1, 0, 2)  # (batch, seq, d_model)
        z = self.encoder(x, src_key_padding_mask=padding_mask)  # (batch, seq, d_model)
        z = z * padding_mask.logical_not().unsqueeze(2)  # (batch, seq, d_model)
        z = z.sum(dim=1) / seq_len.unsqueeze(1)  # (batch, d_model)
        
        # Reparameterization trick
        mu = self.mu(z)  # (batch, d_embed)
        log_std = self.log_std(z)  # (batch, d_embed)
        eps = torch.randn_like(log_std)  # (batch, d_embed)
        z = mu + eps*log_std.exp()  # (batch, d_embed)
        print("mu", mu, flush=True)
        print("log_std", log_std, flush=True)
        print("z", z, flush=True)
        z = self.make_memory(z)  # (batch, d_model)
        z = z.unsqueeze(1)  # (batch, 1, d_model)

        x = self.decoder(
            x,
            z,
            tgt_mask=self.generate_square_subsequent_mask(x.size(dim=1)).to(x.device),
            tgt_key_padding_mask=padding_mask,
            )  # (batch, seq, d_model)
        x = self.classifier(x)  # (batch, seq, vocab_size)
        return x, self.kl_loss(mu, log_std)
    
    @torch.inference_mode()
    def encode(
        self,
        x: torch.Tensor,
        padding_mask: torch.BoolTensor,
        seq_len,
    ):
        """
        padding_mask:
            True is pad
            False is other
        """
        self.eval()
        x = self.embed(x)  # (batch, seq, d_model)
        x = x.permute(1, 0, 2)  # (seq, batch, d_model)
        x = self.pos_encoder(x)  # (seq, batch, d_model)
        x = x.permute(1, 0, 2)  # (batch, seq, d_model)
        z = self.encoder(x, src_key_padding_mask=padding_mask)  # (batch, seq, d_model)
        z = z * padding_mask.logical_not().unsqueeze(2)  # (batch, seq, d_model)
        z = z.sum(dim=1) / seq_len.unsqueeze(1)  # (batch, d_model)

        # Reparameterization trick
        z = self.mu(z)  # (batch, d_embed)
        return z

    def kl_loss(self, mu: torch.Tensor, log_std: torch.Tensor):
        var = log_std.exp()**2
        return torch.sum(-1/2*torch.sum(1 + var.log() - mu**2 - var, dim=1), dim=0)


class TransformerVAEwithViT(TransformerVAE):
    def __init__(
        self,
        vocab_size: int,
        d_embed: int = 16,
        d_model: int = 768,
        nhead: int = 8,
        num_layers: int = 6,
        activation="gelu",
        layer_norm_eps: float = 1e-5,
        batch_first: bool = True,
        norm_first: bool = True,
        ):
        super().__init__(
            vocab_size,
            d_embed,
            d_model,
            nhead,
            num_layers,
            activation,
            layer_norm_eps,
            batch_first,
            norm_first,
        )
        sys.path.append("../../egs/dino")
        from vision_transformer import VisionTransformer
        self.vit = VisionTransformer(patch_size=8, qkv_bias=True)

        state_dict = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/dino/dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth",
            map_location="cuda",
        )

        self.vit.load_state_dict(state_dict)
        self.vit.eval()
        self.vocab_size = vocab_size
    
    def forward(
        self,
        x: torch.Tensor,
        padding_mask: torch.BoolTensor,
        seq_len,
        image: torch.Tensor,
        use_encoder: bool = True,
        verbose: bool = False,
    ):
        """
        padding_mask:
            True is pad
            False is other
        """
        self.train()
        x = self.embed(x)  # (batch, seq, d_model)
        x = x.permute(1, 0, 2)  # (seq, batch, d_model)
        x = self.pos_encoder(x)  # (seq, batch, d_model)
        x = x.permute(1, 0, 2)  # (batch, seq, d_model)
        if use_encoder:
            z = self.encoder(x, src_key_padding_mask=padding_mask)  # (batch, seq, d_model)
            z = z * padding_mask.logical_not().unsqueeze(2)  # (batch, seq, d_model)
            z = z.sum(dim=1) / seq_len.unsqueeze(1)  # (batch, d_model)

            # Reparameterization trick
            mu = self.mu(z)  # (batch, d_embed)
            if verbose:
                print("mu", mu, flush=True)
            # log_std = self.log_std(z)  # (batch, d_embed)
            log_std = torch.full_like(mu, 0.1).log()
            eps = torch.randn_like(log_std)  # (batch, d_embed)

            # eps*log_std.exp() is the noise adding to the action?
            
            z = mu + eps*log_std.exp()  # (batch, d_embed)
            z = self.make_memory(z)  # (batch, d_model)
        
        with torch.no_grad():
            self.vit.eval()
            y = self.vit(image)
        
        if use_encoder:
            z = torch.stack([y, z], dim=1)  # (batch, 2, d_model)
        else:
            z = torch.stack([y], dim=1)  # (batch, 1, d_model)

        x = self.decoder(
            x,
            z,
            tgt_mask=self.generate_square_subsequent_mask(x.size(dim=1)).to(x.device),
            tgt_key_padding_mask=padding_mask,
            )  # (batch, seq, d_model)
        x = self.classifier(x)  # (batch, seq, vocab_size)
        if use_encoder:
            return x, self.kl_loss(mu, log_std)
        else:
            return x, 0
    
    def decode(
        self,
        start_unit: int,
        end_unit: int,
        action: torch.Tensor = None,
        x=None,
        padding_mask=None,
        seq_len=None,
        image=None,
        max_len: int = 100,
        beam_size: int = 50,
    ):
        """
        from https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning/blob/master/eval.py
        """
        self.eval()
        with torch.no_grad():
            if action is not None:
                if action.size(-1) > 768:
                    m_z = self.make_memory(action[:,-self.d_embed:])  # (1, d_model)
                    m = torch.stack([action[:,:-self.d_embed], m_z], dim=1)  # (1, 2, d_model)
                    m = m.expand(beam_size, 2, 768)
                else:
                    m = action.expand(beam_size, 1, 768)
            elif x is None:
                m = self.vit(image)  # (1, d_model)
                m = torch.stack([m], dim=1)  # (1, 1, d_model)
                m = m.expand(beam_size, 1, 768)
            else:
                x = self.embed(x)  # (batch, seq, d_model)
                x = x.permute(1, 0, 2)  # (seq, batch, d_model)
                x = self.pos_encoder(x)  # (seq, batch, d_model)
                x = x.permute(1, 0, 2)  # (batch, seq, d_model)
                
                z = self.encoder(x, src_key_padding_mask=padding_mask)  # (batch, seq, d_model)
                z = z * padding_mask.logical_not().unsqueeze(2)  # (batch, seq, d_model)
                z = z.sum(dim=1) / seq_len.unsqueeze(1)  # (batch, d_model)
                
                # Reparameterization trick
                mu = self.mu(z)  # (batch, d_embed)
                print(mu, flush=True)
                # log_std = self.log_std(z)  # (batch, d_embed)
                log_std = torch.full_like(mu, 0.1).log()
                eps = torch.randn_like(log_std)  # (batch, d_embed)
                z = mu + eps*log_std.exp()  # (batch, d_embed)
                z = self.make_memory(z)  # (batch, d_model)
                
                y = self.vit(image)
                m = torch.stack([y, z], dim=1)  # (batch, 2, d_model)
                m = m.expand(beam_size, 2, y.size(dim=-1))

            k = beam_size
            # Tensor to store top k previous words at each step; now they're just <start>
            k_prev_words = torch.LongTensor([[start_unit]] * k).to(m.device)  # (k, 1)
            # Tensor to store top k sequences; now they're just <start>
            seqs = k_prev_words  # (k, 1)
            # Tensor to store top k sequences' scores; now they're just 0
            top_k_scores = torch.zeros(k, 1).to(m.device)  # (k, 1)
            # Lists to store completed sequences and scores
            complete_seqs = list()
            complete_seqs_scores = list()
            # Start decoding
            step = 1
            # s is a number less than or equal to k, because sequences are removed from this process once they hit <end>
            while True:
                x = self.embed(seqs)  # (1, seq, d_model)
                x = x.permute(1, 0, 2)  # (seq, 1, d_model)
                x = self.pos_encoder(x)  # (seq, 1, d_model)
                x = x.permute(1, 0, 2)  # (1, seq, d_model)
                x = self.decoder(
                    x,
                    m,
                    tgt_mask=self.generate_square_subsequent_mask(x.size(dim=1)).to(x.device),
                    )  # (1, seq, d_model)
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
                m = m[prev_word_inds[incomplete_inds]]
                top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
                k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)
                # Break if things have been going on too long
                if step > max_len:
                    break
                step += 1
            
            if len(complete_seqs_scores) != 0:
                i = complete_seqs_scores.index(max(complete_seqs_scores))
                seq = complete_seqs[i]
            else:
                seq = []
            return seq


class ResNet50(torch.nn.Module):
    def __init__(self):
        super().__init__()
        from torchvision.models.resnet import resnet50
        self.cnn = resnet50(pretrained=False)
        self.cnn.fc = torch.nn.Identity()
        state_dict = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/dino/dino_resnet50_pretrain/dino_resnet50_pretrain.pth",
            map_location="cuda",
        )
        self.cnn.load_state_dict(state_dict, strict=False)
        self.cnn.eval()
        self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7)) # added to ensure 7*7 feature
    
    def forward(self, x):
        x = self.cnn.conv1(x)
        x = self.cnn.bn1(x)
        x = self.cnn.relu(x)
        x = self.cnn.maxpool(x)

        x = self.cnn.layer1(x)
        x = self.cnn.layer2(x)
        x = self.cnn.layer3(x)
        x = self.cnn.layer4(x)
        
        x = self.adaptive_pool(x) # added to ensure 7*7 feature
        
        x = x.permute(0, 2, 3, 1)
        batch, _, _, channels = x.size()
        x = x.view(batch, -1, channels)
        return x


class TransformerVAEwithCNN(TransformerVAE):
    def __init__(
        self,
        vocab_size: int,
        d_embed: int = 16,
        d_model: int = 2048,
        nhead: int = 2048//64,
        num_layers: int = 6,
        activation="gelu",
        layer_norm_eps: float = 1e-5,
        batch_first: bool = True,
        norm_first: bool = True,
        max_len: int = 150,
        ):
        super().__init__(
            vocab_size,
            d_embed,
            d_model,
            nhead,
            num_layers,
            activation,
            layer_norm_eps,
            batch_first,
            norm_first,
        )
        self.pos_encoder=PositionalEncoding(d_model=d_model, max_len=max_len+2)
        self.cnn = ResNet50()
        self.vocab_size = vocab_size
    
    def forward(
        self,
        x: torch.Tensor,
        padding_mask: torch.BoolTensor,
        seq_len,
        image: torch.Tensor,
        use_encoder: bool = True,
        verbose: bool = False,
    ):
        """
        padding_mask:
            True is pad
            False is other
        """
        self.train()
        x = self.embed(x)  # (batch, seq, d_model)
        x = x.permute(1, 0, 2)  # (seq, batch, d_model)
        x = self.pos_encoder(x)  # (seq, batch, d_model)
        x = x.permute(1, 0, 2)  # (batch, seq, d_model)
        if use_encoder:
            z = self.encoder(x, src_key_padding_mask=padding_mask)  # (batch, seq, d_model)
            z = z * padding_mask.logical_not().unsqueeze(2)  # (batch, seq, d_model)
            z = z.sum(dim=1) / seq_len.unsqueeze(1)  # (batch, d_model)

            # Reparameterization trick
            mu = self.mu(z)  # (batch, d_embed)
            if verbose:
                print("mu", mu, flush=True)
            # log_std = self.log_std(z)  # (batch, d_embed)
            log_std = torch.full_like(mu, 0.1).log()
            eps = torch.randn_like(log_std)  # (batch, d_embed)
            z = mu + eps*log_std.exp()  # (batch, d_embed)
            z = self.make_memory(z)  # (batch, d_model)
        
        with torch.no_grad():
            self.cnn.eval()
            y = self.cnn(image)  # (batch, 49, 2048)
        
        if use_encoder:
            z = torch.cat([y, z.unsqueeze(1)], dim=1)  # (batch, 2, d_model)
        else:
            # z = torch.stack([y], dim=1)  # (batch, 1, d_model)
            z = y

        x = self.decoder(
            x,
            z,
            tgt_mask=self.generate_square_subsequent_mask(x.size(dim=1)).to(x.device),
            tgt_key_padding_mask=padding_mask,
            )  # (batch, seq, d_model)
        x = self.classifier(x)  # (batch, seq, vocab_size)
        if use_encoder:
            return x, self.kl_loss(mu, log_std)
        else:
            return x, 0
    
    @torch.inference_mode()
    def decode(
        self,
        start_unit: int,
        end_unit: int,
        action: torch.Tensor = None,
        x=None,
        padding_mask=None,
        seq_len=None,
        image=None,
        max_len: int = 100,
        beam_size: int = 50,
    ):
        """
        from https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning/blob/master/eval.py
        """
        self.eval()
        if action is not None:
            if action.size(-1) > 49*2048:
                fmap = action[:, :49*2048].view(1, 49, 2048)
                embed = action[:, 49*2048:]
                embed = self.make_memory(embed)
                embed = embed.unsqueeze(1)
                m = torch.cat([fmap, embed], dim=1)  # (1, 50, d_model)
                m = m.expand(beam_size, 50, 2048)
            else:
                fmap = action[:, :49*2048].view(1, 49, 2048)
                m = fmap.expand(beam_size, 49, 2048)
        elif x is None:
            m = self.cnn(image)  # (1, d_model)
            m = m.expand(beam_size, 49, 2048)

        k = beam_size
        # Tensor to store top k previous words at each step; now they're just <start>
        k_prev_words = torch.LongTensor([[start_unit]] * k).to(m.device)  # (k, 1)
        # Tensor to store top k sequences; now they're just <start>
        seqs = k_prev_words  # (k, 1)
        # Tensor to store top k sequences' scores; now they're just 0
        top_k_scores = torch.zeros(k, 1).to(m.device)  # (k, 1)
        # Lists to store completed sequences and scores
        complete_seqs = list()
        complete_seqs_scores = list()
        # Start decoding
        step = 1
        # s is a number less than or equal to k, because sequences are removed from this process once they hit <end>
        while True:
            x = self.embed(seqs)  # (1, seq, d_model)
            x = x.permute(1, 0, 2)  # (seq, 1, d_model)
            x = self.pos_encoder(x)  # (seq, 1, d_model)
            x = x.permute(1, 0, 2)  # (1, seq, d_model)
            x = self.decoder(
                x,
                m,
                tgt_mask=self.generate_square_subsequent_mask(x.size(dim=1)).to(x.device),
                )  # (1, seq, d_model)
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
            m = m[prev_word_inds[incomplete_inds]]
            top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
            k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)
            # Break if things have been going on too long
            if step > max_len:
                break
            step += 1
        
        if len(complete_seqs_scores) != 0:
            i = complete_seqs_scores.index(max(complete_seqs_scores))
            seq = complete_seqs[i]
        else:
            seq = []
        return seq