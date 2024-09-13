import torch
import torch.nn as nn
import math
from functools import reduce
from operator import mul
import copy
from semilearn.nets.vit.vit import vit_small_patch16_224



class VPT(nn.Module):
    def __init__(self, vpt_len, seq_len, patch_size, emb_dim, dtype=None):
        super().__init__()
        self.seq_len = seq_len
        self.prompt = nn.Parameter(torch.empty(vpt_len, emb_dim, dtype=dtype))
        init_val = math.sqrt(6. / float(3 * reduce(mul, patch_size, 1) + emb_dim))
        nn.init.uniform_(self.prompt, -init_val, init_val)

    def forward(self, x):
        x = x[:, :self.seq_len, :]
        prompt = self.prompt.expand(x.shape[0], -1, -1)
        x = torch.cat([x, prompt], dim=1)
        return x


class ViT_Tuner(nn.Module):
    def __init__(self, cfg, clip_model, text_features):
        super().__init__()
        n_layers = clip_model.visual.transformer.layers
        emb_dim = clip_model.visual.transformer.width
        seq_len = clip_model.visual.positional_embedding.shape[0]
        patch_size = clip_model.visual.conv1.kernel_size
        dtype = clip_model.dtype

        use_vpt_shallow = cfg.vpt_shallow
        use_vpt_deep = cfg.vpt_deep
        use_vpt_last = cfg.vpt_last

        vpt_len = cfg.vpt_len
        partial = None
        block_num = None



        block_list = []
        if block_num is not None:
            block_num = int(block_num)
            init_block = n_layers - 1
            for i in range(block_num):
                block_list.append(init_block - i)


        assert int(use_vpt_shallow) + int(use_vpt_deep) < 2
        if use_vpt_shallow:
            vpt_list = nn.ModuleList([
                VPT(vpt_len=vpt_len, seq_len=seq_len, patch_size=patch_size, emb_dim=emb_dim, dtype=dtype),
                *[None] * (n_layers - 1)
            ])
        elif use_vpt_deep:
            vpt_list = nn.ModuleList([
                *[None] * (n_layers - partial),
                *[VPT(vpt_len=vpt_len, seq_len=seq_len, patch_size=patch_size, emb_dim=emb_dim, dtype=dtype) for _ in range(partial)]
            ])
        elif use_vpt_last:
            vpt_list = nn.ModuleList([
                *[None] * (n_layers - 1),
                VPT(vpt_len=vpt_len, seq_len=seq_len, patch_size=patch_size, emb_dim=emb_dim, dtype=dtype)
            ])
        else:
            vpt_list = [None] * n_layers


        # To be optimized
        self.vpt_list = vpt_list

        self.proj = copy.deepcopy(clip_model.visual.proj)
        self.logit_scale = nn.Parameter(torch.ones([]) * (1 / 0.07))

        self.alpha_mat = nn.Parameter(torch.ones((cfg.DATA.NUMBER_CLASSES, cfg.DATA.NUMBER_CLASSES), dtype=dtype) * cfg.alpha)
        self.text_emb = nn.Parameter(text_features.clone())

        self.alpha_cls = nn.Parameter(torch.ones((cfg.DATA.NUMBER_CLASSES, ), dtype=dtype) * cfg.alpha)
        self.alpha = nn.Parameter(torch.ones([]) * cfg.alpha)


class Model(nn.Module):
    def __init__(self, cfg, clip_model, text_features):
        super().__init__()
        self.image_encoder = vit_small_patch16_224()
        self.tuner = ViT_Tuner(cfg, clip_model, text_features)

    def forward(self, image):
        feat = self.image_encoder(image, self.tuner)
        return feat


