# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Zhenda Xie
# --------------------------------------------------------

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from diffdist import functional


def dist_collect(x):
    """ collect all tensor from all GPUs
    args:
        x: shape (mini_batch, ...)
    returns:
        shape (mini_batch * num_gpu, ...)
    """
    x = x.contiguous()
    out_list = [torch.zeros_like(x, device=x.device, dtype=x.dtype).contiguous()
                for _ in range(dist.get_world_size())]
    out_list = functional.all_gather(out_list, x)
    return torch.cat(out_list, dim=0).contiguous()


# TODO: define covariance contrastive loss function based on the paper of "TiCO: Transformation Invariance
# TODO: Contrastive Learning of Visual Representations"

class WDA_Loss(nn.Module):
    def __init__(self, beta, rho, gamma):
        super().__init__()

        self.beta = beta
        self.rho = rho
        self.gamma = gamma

    def forward(self, C, q, k):
        B = torch.mm(q.T, q)/q.shape[0]
        C = self.beta * C + (1 - self.beta) * B
        trans_inv = -(q.T @ k).sum(dim=1).mean() + self.rho * \
            (torch.mm(k.T, C) @ k).sum(dim=1).mean()
        return trans_inv, C


class covariance_Loss(nn.Module):
    def __init__(self, alpha):
        super().__init__()
        self.alpha = alpha

    def forward(self, k, k_large):
        return -1 * self.alpha * (k.T @ k_large).sum(dim=1).mean()


class WDASSL(nn.Module):
    def __init__(self,
                 cfg,
                 encoder,
                 encoder_k,
                 contrast_momentum=0.99,
                 contrast_temperature=0.2,
                 contrast_num_negative=4096,
                 proj_num_layers=2,
                 pred_num_layers=2,
                 **kwargs):
        super().__init__()

        self.cfg = cfg

        self.encoder = encoder
        self.encoder_k = encoder_k

        self.contrast_momentum = contrast_momentum
        self.contrast_temperature = contrast_temperature
        self.contrast_num_negative = contrast_num_negative

        self.proj_num_layers = proj_num_layers
        self.pred_num_layers = pred_num_layers

        self.projector = WdasslMLP(
            in_dim=self.encoder.num_features, num_layers=proj_num_layers)
        self.projector_k = WdasslMLP(
            in_dim=self.encoder.num_features, num_layers=proj_num_layers)
        self.predictor = WdasslMLP(num_layers=pred_num_layers)

        for param_q, param_k in zip(self.encoder.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        for param_q, param_k in zip(self.projector.parameters(), self.projector_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        if self.cfg.MODEL.WDA.NORM_BEFORE_MLP == 'bn':
            nn.SyncBatchNorm.convert_sync_batchnorm(self.encoder)
            nn.SyncBatchNorm.convert_sync_batchnorm(self.encoder_k)

        nn.SyncBatchNorm.convert_sync_batchnorm(self.projector)
        nn.SyncBatchNorm.convert_sync_batchnorm(self.projector_k)
        nn.SyncBatchNorm.convert_sync_batchnorm(self.predictor)

        self.K = int(self.cfg.DATA.TRAINING_IMAGES * 1. / dist.get_world_size() /
                     self.cfg.DATA.BATCH_SIZE) * self.cfg.TRAIN.EPOCHS
        self.k = int(self.cfg.DATA.TRAINING_IMAGES * 1. / dist.get_world_size() /
                     self.cfg.DATA.BATCH_SIZE) * self.cfg.TRAIN.START_EPOCH

        # create the queue
        self.register_buffer("queue1", torch.randn(
            256, self.contrast_num_negative))
        self.register_buffer("queue2", torch.randn(
            256, self.contrast_num_negative))
        self.queue1 = F.normalize(self.queue1, dim=0)
        self.queue2 = F.normalize(self.queue2, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        # self.C_prev = torch.Variable(torch.zeros(
        #     self.predictor.out_dim, self.predictor.out_dim))
        # self.C_prev = self.C_prev.detach()

        # self.WDAloss = WDA_Loss(beta=0.99, rho=0.99, gamma=0.99)
        # self.covariance_loss = covariance_Loss(alpha=0.99)

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        _contrast_momentum = 1. - \
            (1. - self.contrast_momentum) * \
            (np.cos(np.pi * self.k / self.K) + 1) / 2.
        self.k = self.k + 1

        for param_q, param_k in zip(self.encoder.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * _contrast_momentum + \
                param_q.data * (1. - _contrast_momentum)

        for param_q, param_k in zip(self.projector.parameters(), self.projector_k.parameters()):
            param_k.data = param_k.data * _contrast_momentum + \
                param_q.data * (1. - _contrast_momentum)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys1, keys2):
        # gather keys before updating queue
        keys1 = dist_collect(keys1)
        keys2 = dist_collect(keys2)

        batch_size = keys1.shape[0]

        ptr = int(self.queue_ptr)
        assert self.contrast_num_negative % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue1[:, ptr:ptr + batch_size] = keys1.T
        self.queue2[:, ptr:ptr + batch_size] = keys2.T
        ptr = (ptr + batch_size) % self.contrast_num_negative  # move pointer

        self.queue_ptr[0] = ptr

    def contrastive_loss(self, q, k, queue):

        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q, queue.clone().detach()])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.contrast_temperature

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        return F.cross_entropy(logits, labels)

    def forward(self, im_1, im_2):
        feat_1, feat_1_large = self.encoder(im_1)  # queries: NxC
        proj_1 = self.projector(feat_1)
        pred_1 = self.predictor(proj_1)
        pred_1 = F.normalize(pred_1, dim=1)

        # for large window
        proj_1_large = self.projector(feat_1_large)
        pred_1_large = self.predictor(proj_1_large)
        pred_1_large = F.normalize(pred_1_large, dim=1)

        feat_2, feat_2_large = self.encoder(im_2)
        proj_2 = self.projector(feat_2)
        pred_2 = self.predictor(proj_2)
        pred_2 = F.normalize(pred_2, dim=1)

        # for large window
        proj_2_large = self.projector(feat_2_large)
        pred_2_large = self.predictor(proj_2_large)
        pred_2_large = F.normalize(pred_2_large, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            feat_1_ng, feat_1_ng_large = self.encoder_k(im_1)  # keys: NxC

            # print(feat_1_ng)

            proj_1_ng = self.projector_k(feat_1_ng)
            proj_1_ng = F.normalize(proj_1_ng, dim=1)

            # for large window
            proj_1_ng_large = self.projector_k(feat_1_ng_large)
            proj_1_ng_large = F.normalize(proj_1_ng_large, dim=1)

            feat_2_ng, feat_2_ng_large = self.encoder_k(im_2)
            proj_2_ng = self.projector_k(feat_2_ng)
            proj_2_ng = F.normalize(proj_2_ng, dim=1)

            # for large window
            proj_2_ng_large = self.projector_k(feat_2_ng_large)
            proj_2_ng_large = F.normalize(proj_2_ng_large, dim=1)
            # print((pred_1_large @ proj_2_ng_large.transpose(0, 1) /
            #        torch.norm(pred_1_large @ proj_2_ng_large.transpose(0, 1), dim=1, keepdim=True)).shape)

            # normalize the key features with covariance matrix between pred_1_large and proj_2_ng_large
            proj_1_ng = (pred_1_large @ proj_2_ng_large.transpose(0, 1) /
                         torch.norm(pred_1_large @ proj_2_ng_large.transpose(0, 1), dim=1, keepdim=True)) @ proj_1_ng

            # normalize the key features with covariance matrix between pred_2_large and proj_1_ng_large
            proj_2_ng = (pred_2_large @ proj_1_ng_large.transpose(0, 1) /
                         torch.norm(pred_2_large @ proj_1_ng_large.transpose(0, 1), dim=1, keepdim=True)) @ proj_2_ng

        loss = self.contrastive_loss(pred_1, proj_2_ng, self.queue2) \
            + self.contrastive_loss(pred_2, proj_1_ng, self.queue1)

        self._dequeue_and_enqueue(proj_1_ng, proj_2_ng)

        return loss


class WdasslMLP(nn.Module):
    def __init__(self, in_dim=256, inner_dim=4096, out_dim=256, num_layers=2):
        super(WdasslMLP, self).__init__()

        # hidden layers
        linear_hidden = [nn.Identity()]
        for i in range(num_layers - 1):
            linear_hidden.append(
                nn.Linear(in_dim if i == 0 else inner_dim, inner_dim))
            linear_hidden.append(nn.BatchNorm1d(inner_dim))
            linear_hidden.append(nn.ReLU(inplace=True))
        self.linear_hidden = nn.Sequential(*linear_hidden)

        self.linear_out = nn.Linear(
            in_dim if num_layers == 1 else inner_dim, out_dim) if num_layers >= 1 else nn.Identity()

    def forward(self, x):
        x = self.linear_hidden(x)
        x = self.linear_out(x)

        return x
