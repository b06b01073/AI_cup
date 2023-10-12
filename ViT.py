import torch.nn as nn
import torch
from einops.layers.torch import Rearrange, Reduce
from einops import repeat, rearrange
import torch.nn.functional as F


class LinearProj(nn.Module):
    def __init__(self, patch_size, in_channels, embedded_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, embedded_dim, kernel_size=patch_size, stride=patch_size),
            Rearrange('b e h w -> b (h w) e')
        )

    def forward(self, x):
        x = self.net(x)
        return x


class PatchEmbedder(nn.Module):
    def __init__(self, num_patches, embedded_dim, drop):
        super().__init__()
        self.cls_token = nn.Parameter(torch.rand(1, embedded_dim))
        self.pos_embedding = nn.Parameter(torch.rand(num_patches + 1, embedded_dim))
        self.dropout = nn.Dropout(p=drop)
    
    def forward(self, x):
        batch_size = x.shape[0]
        cls = repeat(self.cls_token, 'n e -> b n e', b=batch_size)
        pos_emb = repeat(self.pos_embedding, 'n e -> b n e', b=batch_size)

        x = torch.cat([cls, x], dim=1)
        x = x + pos_emb


        return x


class TranformerEncoder(nn.Module):
    def __init__(self, embedded_dim, hidden_dim, num_head, drop):
        super().__init__()
        self.net = nn.Sequential(
            EncoderSublayer(MultiAtt(num_head, embedded_dim, drop), embedded_dim),
            nn.Dropout(p=drop),
            EncoderSublayer(MLP(embedded_dim, hidden_dim, drop), embedded_dim),
            nn.Dropout(p=drop),
        )

    def forward(self, x):
        return self.net(x)


class EncoderSublayer(nn.Module):
    def __init__(self, sublayer, embedded_dim):
        super().__init__()
        self.layer_norm = nn.LayerNorm(embedded_dim)
        self.sublayer = sublayer

    def forward(self, x):
        x = x + self.sublayer(x)
        x = self.layer_norm(x)
        
        return x


class MultiAtt(nn.Module):
    def __init__(self, heads, embedded_dim, drop):
        super().__init__()
        self.num_head = heads
        self.scaling = (embedded_dim // heads) ** (-0.5)

        self.pre_proj = nn.Linear(embedded_dim, embedded_dim * 3) # project the input into 3 * embedded_dim, and unpack them into q, k, v
        self.proj = nn.Linear(embedded_dim, embedded_dim)
        self.dropout = nn.Dropout(p=drop)
        

    def forward(self, x):
        x = rearrange(self.pre_proj(x), 'b n (u h e) -> u b h n e', u=3, h=self.num_head)
        q, k, v = x[0], x[1], x[2]

        similarity = torch.einsum('bhqi, bhki -> bhqk', q, k) * self.scaling
        similarity = F.softmax(similarity, dim=-1) 

        att_score = torch.einsum('bhnk, bhke -> bhne', similarity, v)
        att_score = rearrange(att_score, 'b h n e -> b n (h e)')

        att_score = self.dropout(att_score)
        att_score = self.proj(att_score)

        return att_score



class MLP(nn.Module):
    def __init__(self, embedded_dim, hidden_dim, drop):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedded_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(p=drop),
            nn.Linear(hidden_dim, embedded_dim)
        )

    def forward(self, x):
        return self.net(x)


class Classifier(nn.Module):
    def __init__(self, embedded_dim, num_class, drop):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(embedded_dim),
            nn.Dropout(p=drop),
            nn.Linear(embedded_dim, num_class)
        )

    def forward(self, x):
        return self.net(x)


class ViT(nn.Module):
    def __init__(self, img_shape, patch_size, embedded_dim, encoder_layers, num_class, num_head, drop):
        super().__init__()
        c, h, w = img_shape
        self.num_patch = int((h // patch_size) * (w // patch_size))
        encoder_layers = [TranformerEncoder(embedded_dim, embedded_dim*4, num_head, drop) for _ in range(encoder_layers)]

        self.net = nn.ModuleList([
            LinearProj(patch_size, c, embedded_dim),
            PatchEmbedder(self.num_patch, embedded_dim, drop),
            *encoder_layers,
        ])
        self.output_layer = Classifier(embedded_dim, num_class, drop)


    def forward(self, x):
        for layer in self.net:
            x = layer(x)
            # print(x.shape)
        
        cls_token = x[:, 0]
        output = self.output_layer(cls_token)

        return output