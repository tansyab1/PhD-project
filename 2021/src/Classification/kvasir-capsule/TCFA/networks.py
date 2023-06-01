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

        self.training = training

        # create an autoencoder block for input size 224*224*3 containing 2 conv layers
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.cross_attention_layer = CrossAttention(
            dim=self.attention_dim,
            heads=8,
            dim_head=64,
            dropout=0.,
            patch_size_large=16,
            input_size=56,
            channels=64)

        self.self_attention_layer = selfImagEmbedder(
            dim=self.attention_dim,
            heads=8,
            dim_head=64,
            dropout=0.,
            patch_size_large=16,
            input_size=56,
            channels=64)

        self.self_attention_layer2 = selfImagEmbedder(
            dim=self.attention_dim,
            heads=8,
            dim_head=64,
            dropout=0.,
            patch_size_large=16,
            input_size=56,
            channels=64)

        # MLP to encode the image to extract the noise level
        self.encoder_mlp = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=3, stride=2),
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

        cross_attention_feature = cross_attention_feature.reshape(
            -1, self.dim, self.shape[0], self.shape[1])

        encoded_noise_attention = torch.mul(
            encoded_noise, cross_attention_feature)

        encoded_image = torch.cat(
            (encoded_noise_attention, encoded_image), dim=1)

        decoded_image = self.decoder(encoded_image)

        self.base_model = self.base_model.to('cuda:1')
        decoded_image = decoded_image.to('cuda:1')
        reference = reference.to('cuda:1')

        resnet_out = self.base_model(decoded_image)
        if self.training:
            resnet_out_encoded = self.base_model(reference)
            return (resnet_out,
                    resnet_out_encoded,
                    decoded_image, encoded_noise,
                    encoded_positive,
                    encoded_negative,
                    mu,
                    logvar)
        else:
            return decoded_image


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

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches+1, dim))

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
        super(ImageEmbedder, self).__init__()
        num_patches = (image_size // patch_size) ** 2
        self.to_patch_embedding_x = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)',
                      p1=patch_size, p2=patch_size),
            nn.Linear(channels*patch_size*patch_size, dim),
        )
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches+1, dim))

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
                 input_size=56,
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

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, channels*patch_size_large*patch_size_large),
            Rearrange('b (h w) (p1 p2 c) -> b c (h p1) (w p2)',
                      p1=patch_size_large,
                      p2=patch_size_large,
                      c=channels,
                      h=num_patches_large),  # 14x14 patches with c = 128
        )

        self.matrix = nn.Linear(dim_head, num_patches_large*num_patches_large)

    def forward(self, x_q):
        x_q, x_kv = self.image_embedder(x_q, x_q)
        b, n_q, _ = x_q.shape
        b, n_kv, _ = x_kv.shape

        k = self.to_k(x_kv)
        v = self.to_v(x_kv)
        q = self.to_q(x_q)

        k = rearrange(k, 'b n (h d) -> b h n d', h=self.heads)
        v = rearrange(v, 'b n (h d) -> b h n d', h=self.heads)
        q = rearrange(q, 'b n (h d) -> b h n d', h=self.heads)

        sim = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = sim.softmax(dim=-1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)', h=self.heads)
        out = self.to_out(out)
        out = self.mlp_head(out)
        out = self.matrix(out)
        return out


class CrossAttention(nn.Module):
    def __init__(self, dim=2048,
                 heads=8,
                 dim_head=256,
                 dropout=0.,
                 patch_size_large=7,
                 input_size=56,
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

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, channels*patch_size_large*patch_size_large),
            Rearrange('b (h w) (p1 p2 c) -> b c (h p1) (w p2)',
                      p1=patch_size_large,
                      p2=patch_size_large,
                      c=channels,
                      h=num_patches_large),  # 14x14 patches with c = 128
        )

        self.matrix = nn.Linear(dim_head, num_patches_large*num_patches_large)

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

        b, n, _, h = *x_q.shape, self.heads

        k = self.to_k(x_kv_token)
        k = rearrange(k, 'b n (h d) -> b h n d', h=h, b=b, n=n)

        v = self.to_v(x_kv_token)
        v = rearrange(v, 'b n (h d) -> b h n d', h=h, b=b, n=n)

        q = self.to_q(x_q_token)
        q = rearrange(q, 'b n (h d) -> b h n d', h=h, b=b, n=n)

        dots = self.reparameterize(k, q) * self.scale
        dots = rearrange(dots, 'b h n d -> b (h n) d')
        dots = self.matrix(dots)
        dots = rearrange(dots, 'b (h i) d -> b h i d', b=b, h=h)
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
