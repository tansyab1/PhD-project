from __future__ import print_function, division

import torch
import torch.nn as nn
from torchvision import models
from einops import rearrange
from einops.layers.torch import Rearrange
from torch import einsum

# ================================================
# New architecture
# ================================================


class MyNet(nn.Module):
    def __init__(self, num_out=14, training=True):
        super(MyNet, self).__init__()
        self.shape = (336, 336)
        self.flatten_dim = self.shape[0]/4 * self.shape[1]/4 * 64
        self.attention_dim = 2048

        self.base_model = BaseNet(num_out)
        for param in self.base_model.parameters():
            param.requires_grad = False

        # self.training = training

        # create an autoencoder block for input size 224*224*3 containing 2 conv layers
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.cross_attention_layer = CrossAttention(
            dim=self.attention_dim,
            heads=8,
            dim_head=64,
            dropout=0.,
            patch_size_large=14,
            input_size=84,
            channels=64)

        self.self_attention_layer = SelfAttention(
            dim=self.attention_dim,
            heads=8,
            dim_head=64,
            dropout=0.,
            patch_size_large=14,
            input_size=84,
            channels=64)

        self.self_attention_layer2 = SelfAttention(
            dim=self.attention_dim,
            heads=8,
            dim_head=64,
            dropout=0.,
            patch_size_large=14,
            input_size=84,
            channels=64)

        # MLP to encode the image to extract the noise level
        self.encoder_mlp = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3,
                               stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=3, stride=2,
                               padding=1, output_padding=1),
            nn.Tanh(),
        )

    def forward(self, x, positive, negative, reference):
        x = x.cuda(1)
        self.encoder = self.encoder.cuda(1)
        encoded_image = self.encoder(x)
        x = x.cuda(0)
        encoded_image = encoded_image.cuda(0)
        # encode the image with channel = 64
        inputs = [x, positive, negative]
        encoded_inputs = [self.encoder_mlp(input) for input in inputs]
        encoded_noise, encoded_positive, encoded_negative = encoded_inputs

        encoded_image = self.self_attention_layer(encoded_image)
        encoded_noise = self.self_attention_layer2(encoded_noise)

        cross_attention_feature, mu, logvar = self.cross_attention_layer(
            encoded_image, encoded_noise)

        encoded_noise_attention = torch.mul(
            encoded_noise, cross_attention_feature)

        encoded_image = torch.cat(
            (encoded_noise_attention, encoded_image), dim=1)

        decoded_image = self.decoder(encoded_image)

        self.base_model = self.base_model.to('cuda:1')
        decoded_image = decoded_image.to('cuda:1')
        reference = reference.to('cuda:1')

        resnet_out = self.base_model(decoded_image)
        resnet_out_encoded = self.base_model(reference)
        return resnet_out, resnet_out_encoded, decoded_image, encoded_noise, encoded_positive, encoded_negative, mu, logvar


class BaseNet(nn.Module):
    def __init__(self, num_out=14):
        super(BaseNet, self).__init__()
        self.resnet_model = models.densenet121(pretrained=True)
        self.module = nn.Sequential(*list(self.resnet_model.children())[:-1])

    def forward(self, x):
        x = self.module(x)
        return x

# self.cross_attention_layer = CrossAttention(
#           dim=self.attention_dim, heads=7, dim_head=32, dropout=0., patch_size_large=16, input_size=224, channels=64)


class ImageEmbedder(nn.Module):
    def __init__(self, dim, image_size, patch_size, channels):
        super(ImageEmbedder, self).__init__()
        num_patches = (image_size // patch_size) ** 2
        self.to_patch_embedding_x = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)',
                      p1=patch_size, p2=patch_size),
            nn.Linear(channels*patch_size*patch_size, dim),
        )

        self.to_patch_embedding_noise = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)',
                      p1=patch_size, p2=patch_size),
            nn.Linear(channels*patch_size*patch_size, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))

    def forward(self, x_q, x_kv):
        x_q = self.to_patch_embedding_x(x_q)
        x_kv = self.to_patch_embedding_noise(x_kv)
        x_q = torch.cat((self.pos_embedding.repeat(
            x_q.shape[0], 1, 1), x_q), dim=1)
        x_kv = torch.cat((self.pos_embedding.repeat(
            x_kv.shape[0], 1, 1), x_kv), dim=1)
        return x_q, x_kv


class selfImagEmbedder(nn.Module):
    def __init__(self, dim, image_size, patch_size, channels):
        super(selfImagEmbedder, self).__init__()
        num_patches = (image_size // patch_size) ** 2
        # print("num_patches", num_patches)
        self.to_patch_embedding_x = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)',
                      p1=patch_size, p2=patch_size),
            nn.Linear(channels*patch_size*patch_size, dim),
        )
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))

    def forward(self, x_q):
        x_q = self.to_patch_embedding_x(x_q)
        x_q = torch.cat((self.pos_embedding.repeat(
            x_q.shape[0], 1, 1), x_q), dim=1)
        return x_q


class SelfAttention(nn.Module):
    def __init__(self, dim=2048,
                 heads=8,
                 dim_head=256,
                 dropout=0.,
                 patch_size_large=7,
                 input_size=84,
                 channels=128):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.image_embedder = selfImagEmbedder(
            dim=dim,
            image_size=input_size,
            patch_size=patch_size_large,
            channels=channels
        )

        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)

        # num_patches_large = (input_size // patch_size_large) ** 2

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, channels*patch_size_large*patch_size_large),
            Rearrange('b n (p1 p2 c) -> b c (n p1 p2)',
                      p1=patch_size_large,
                      p2=patch_size_large,
                      c=channels,),  # 14x14 patches with c = 128
        )

        self.matrix = nn.Sequential(
            nn.Linear(2*input_size*input_size, input_size*input_size),
            Rearrange('b c (p1 p2) -> b c p1 p2',
                      p1=input_size,
                      p2=input_size,),  # 14x14 patches with c = 128
        )

    def forward(self, x_q):
        x_q = self.image_embedder(x_q)
        b, n_q, _ = x_q.shape

        k = self.to_k(x_q)
        v = self.to_v(x_q)
        q = self.to_q(x_q)

        k = rearrange(k, 'b n (h d) -> b h n d', h=self.heads)
        v = rearrange(v, 'b n (h d) -> b h n d', h=self.heads)
        q = rearrange(q, 'b n (h d) -> b h n d', h=self.heads)

        sim = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = sim.softmax(dim=-1)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)', h=self.heads)
        out = self.to_out(out)
        # print("out", out.shape)
        out = self.mlp_head(out)
        # print("out mlp", out.shape)
        out = self.matrix(out)
        # print("out matrix", out.shape)
        return out


class CrossAttention(nn.Module):
    def __init__(self, dim=2048,
                 heads=8,
                 dim_head=256,
                 dropout=0.,
                 patch_size_large=7,
                 input_size=84,
                 channels=128):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.image_embedder = ImageEmbedder(
            dim=dim,
            image_size=input_size,
            patch_size=patch_size_large,
            channels=channels)

        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)

        num_patches_large = (input_size // patch_size_large) ** 2
        self.h = input_size // patch_size_large

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, channels*patch_size_large*patch_size_large),
            # transpose channel 2 and 3
            Rearrange('b n v -> b v n'),
            nn.Linear(2*num_patches_large, num_patches_large),
            # transpose channel 2 and 3
            Rearrange('b v n -> b n v'),
            Rearrange('b (h w) (p1 p2 c) -> b c (h p1) (w p2)',
                      p1=patch_size_large,
                      p2=patch_size_large,
                      c=channels,
                      h=self.h,)
        )

        self.matrix = nn.Linear(dim_head, 2*num_patches_large)

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        std = std.to('cuda')
        # return torch.normal(mu, std)
        esp = torch.randn(*mu.size())
        esp = esp.to('cuda')
        z = mu + std * esp
        return z

    def forward(self, x_q, x_kv):
        x_q_token, x_kv_token = self.image_embedder(x_q, x_kv)

        # b, n, _, h = *x_q.shape, self.heads

        k = self.to_k(x_kv_token)
        k = rearrange(k, 'b n (h d) -> b h n d', h=self.heads)

        v = self.to_v(x_kv_token)
        v = rearrange(v, 'b n (h d) -> b h n d', h=self.heads)

        q = self.to_q(x_q_token)
        q = rearrange(q, 'b n (h d) -> b h n d', h=self.heads)

        dots = self.reparameterize(k, q) * self.scale
        dots = rearrange(dots, 'b h n d -> b (h n) d')
        dots = self.matrix(dots)
        dots = rearrange(dots, 'b (h i) d -> b h i d', h=self.heads)
        attn = dots.softmax(dim=-1)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        out = self.mlp_head(out)
        q = rearrange(q, 'b h n d -> b (h n d)')
        k = rearrange(k, 'b h n d -> b (h n d)')

        # normalize q and k
        q = q / torch.norm(q, dim=1, keepdim=True)
        k = k / torch.norm(k, dim=1, keepdim=True)
        return out, q, k
