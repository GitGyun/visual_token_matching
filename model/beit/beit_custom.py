import math
from functools import partial
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.helpers import build_model_with_cfg
from timm.models.layers import PatchEmbed, DropPath, trunc_normal_
from .beit_registry import register_model
from timm.models.vision_transformer import checkpoint_filter_fn

from einops import repeat


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic', 'fixed_input_size': True,
        'mean': (0.5, 0.5, 0.5), 'std': (0.5, 0.5, 0.5),
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }


default_cfgs = {
    'beit_base_patch16_224': _cfg(
        url='https://unilm.blob.core.windows.net/beit/beit_base_patch16_224_pt22k_ft22kto1k.pth'),
    'beit_base_patch16_384': _cfg(
        url='https://unilm.blob.core.windows.net/beit/beit_base_patch16_384_pt22k_ft22kto1k.pth',
        input_size=(3, 384, 384), crop_pct=1.0,
    ),
    'beit_base_patch16_224_in22k': _cfg(
        url='https://unilm.blob.core.windows.net/beit/beit_base_patch16_224_pt22k_ft22k.pth',
        num_classes=21841,
    ),
    'beit_large_patch16_224': _cfg(
        url='https://unilm.blob.core.windows.net/beit/beit_large_patch16_224_pt22k_ft22kto1k.pth'),
    'beit_large_patch16_384': _cfg(
        url='https://unilm.blob.core.windows.net/beit/beit_large_patch16_384_pt22k_ft22kto1k.pth',
        input_size=(3, 384, 384), crop_pct=1.0,
    ),
    'beit_large_patch16_512': _cfg(
        url='https://unilm.blob.core.windows.net/beit/beit_large_patch16_512_pt22k_ft22kto1k.pth',
        input_size=(3, 512, 512), crop_pct=1.0,
    ),
    'beit_large_patch16_224_in22k': _cfg(
        url='https://unilm.blob.core.windows.net/beit/beit_large_patch16_224_pt22k_ft22k.pth',
        num_classes=21841,
    ),
}


class CustomMlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, n_tasks, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, bias=True, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = (bias, bias)
        drop_probs = (drop, drop)

        self.fc1 = CustomLinear(n_tasks, in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = CustomLinear(n_tasks, hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x, t_idx=None):
        x = self.fc1(x, t_idx)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x, t_idx)
        x = self.drop2(x)
        return x


class CustomLinear(nn.Linear):
    def __init__(self, n_tasks=0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.n_tasks = n_tasks
        if self.n_tasks > 0:
            assert self.bias is not None
            self.bias = nn.Parameter(repeat(self.bias.data, '... -> T ...', T=n_tasks).contiguous())

    def forward(self, input, t_idx=None):
        if self.n_tasks > 0:
            assert t_idx is not None
            output = F.linear(input, self.weight, None)
            return output + self.bias[t_idx][:, None]
        else:
            return F.linear(input, self.weight, self.bias)


class CustomLayerNorm(nn.LayerNorm):
    def __init__(self, n_tasks=0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.n_tasks = n_tasks
        if self.n_tasks > 0:
            assert self.elementwise_affine
            self.bias = nn.Parameter(repeat(self.bias.data, '... -> T ...', T=n_tasks).contiguous())

    def forward(self, input, t_idx=None):
        if self.n_tasks > 0:
            assert t_idx is not None
            output = F.layer_norm(input, self.normalized_shape, self.weight, None, self.eps)
            return output + self.bias[t_idx][:, None]
        else:
            return F.layer_norm(
                input, self.normalized_shape, self.weight, self.bias, self.eps)


class CustomAttention(nn.Module):
    def __init__(
            self, dim, num_heads=8, qkv_bias=False, attn_drop=0.,
            proj_drop=0., window_size=None, attn_head_dim=None, n_prompts=0, n_tasks=0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = head_dim ** -0.5
        self.n_prompts = n_prompts

        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.register_buffer('k_bias', torch.zeros(all_head_dim), persistent=False)
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.k_bias = None
            self.v_bias = None

        if window_size:
            self.window_size = window_size
            self.num_relative_distance = (2 * window_size[0] - 1) * (2 * window_size[1] - 1) + 3
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros(self.num_relative_distance, num_heads))  # 2*Wh-1 * 2*Ww-1, nH
            # cls to token & token 2 cls & cls to cls

            # get pair-wise relative position index for each token inside the window
            coords_h = torch.arange(window_size[0])
            coords_w = torch.arange(window_size[1])
            coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
            coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
            relative_coords[:, :, 0] += window_size[0] - 1  # shift to start from 0
            relative_coords[:, :, 1] += window_size[1] - 1
            relative_coords[:, :, 0] *= 2 * window_size[1] - 1
            relative_position_index = \
                torch.zeros(size=(window_size[0] * window_size[1] + 1,) * 2, dtype=relative_coords.dtype)
            relative_position_index[1:, 1:] = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
            relative_position_index[0, 0:] = self.num_relative_distance - 3
            relative_position_index[0:, 0] = self.num_relative_distance - 2
            relative_position_index[0, 0] = self.num_relative_distance - 1

            self.register_buffer("relative_position_index", relative_position_index)
        else:
            self.window_size = None
            self.relative_position_bias_table = None
            self.relative_position_index = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = CustomLinear(n_tasks, all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, rel_pos_bias: Optional[torch.Tensor] = None, t_idx=None):
        B, N, C = x.shape
        qkv_bias = torch.cat((self.q_bias, self.k_bias, self.v_bias)) if self.q_bias is not None else None
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        if self.relative_position_bias_table is not None:
            relative_position_bias = \
                self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                    self.window_size[0] * self.window_size[1] + 1,
                    self.window_size[0] * self.window_size[1] + 1, -1)  # Wh*Ww,Wh*Ww,nH
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
            
            _C, _H, _W = relative_position_bias.shape
            
            if self.n_prompts > 0:
                relative_position_bias = torch.cat((
                    torch.zeros(_C, self.n_prompts, _W, dtype=attn.dtype, device=attn.device),
                    relative_position_bias
                    ), dim=-2)

                relative_position_bias = torch.cat((
                    torch.zeros(_C, _H + self.n_prompts, self.n_prompts, dtype=attn.dtype, device=attn.device),
                    relative_position_bias
                    ), dim=-1)
            
            attn = attn + relative_position_bias.unsqueeze(0)

        if rel_pos_bias is not None:
            attn = attn + rel_pos_bias

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x, t_idx)
        x = self.proj_drop(x)
        return x


class CustomBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., init_values=None, act_layer=nn.GELU, norm_layer=CustomLayerNorm,
                 window_size=None, attn_head_dim=None, n_prompts=0, n_tasks=0):
        super().__init__()
        self.norm1 = norm_layer(n_tasks, dim)
        self.attn = CustomAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop,
            window_size=window_size, attn_head_dim=attn_head_dim, n_prompts=n_prompts, n_tasks=n_tasks)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(n_tasks, dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = CustomMlp(n_tasks=n_tasks, in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if init_values:
            self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
        else:
            self.gamma_1, self.gamma_2 = None, None

    def forward(self, x, rel_pos_bias: Optional[torch.Tensor] = None, t_idx=None):
        if self.gamma_1 is None:
            x = x + self.drop_path(self.attn(self.norm1(x, t_idx), rel_pos_bias=rel_pos_bias, t_idx=t_idx))
            x = x + self.drop_path(self.mlp(self.norm2(x, t_idx), t_idx))
        else:
            x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x, t_idx), rel_pos_bias=rel_pos_bias, t_idx=t_idx))
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x, t_idx), t_idx))
        return x


class RelativePositionBias(nn.Module):

    def __init__(self, window_size, num_heads):
        super().__init__()
        self.window_size = window_size
        self.num_relative_distance = (2 * window_size[0] - 1) * (2 * window_size[1] - 1) + 3
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(self.num_relative_distance, num_heads))  # 2*Wh-1 * 2*Ww-1, nH
        # cls to token & token 2 cls & cls to cls

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(window_size[0])
        coords_w = torch.arange(window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * window_size[1] - 1
        relative_position_index = \
            torch.zeros(size=(window_size[0] * window_size[1] + 1,) * 2, dtype=relative_coords.dtype)
        relative_position_index[1:, 1:] = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        relative_position_index[0, 0:] = self.num_relative_distance - 3
        relative_position_index[0:, 0] = self.num_relative_distance - 2
        relative_position_index[0, 0] = self.num_relative_distance - 1

        self.register_buffer("relative_position_index", relative_position_index)

        # trunc_normal_(self.relative_position_bias_table, std=.02)

    def forward(self):
        relative_position_bias = \
            self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                self.window_size[0] * self.window_size[1] + 1,
                self.window_size[0] * self.window_size[1] + 1, -1)  # Wh*Ww,Wh*Ww,nH
        return relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww


class CustomBeit(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=partial(CustomLayerNorm, eps=1e-6), init_values=None,
                 use_abs_pos_emb=True, use_rel_pos_bias=False, use_shared_rel_pos_bias=False,
                 use_mean_pooling=True, init_scale=0.001, n_prompts=0, n_tasks=0):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        if use_abs_pos_emb:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        else:
            self.pos_embed = None
        self.pos_drop = nn.Dropout(p=drop_rate)

        if use_shared_rel_pos_bias:
            self.rel_pos_bias = RelativePositionBias(window_size=self.patch_embed.grid_size, num_heads=num_heads)
        else:
            self.rel_pos_bias = None

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.use_rel_pos_bias = use_rel_pos_bias
        self.blocks = nn.ModuleList([
            CustomBlock(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                init_values=init_values, window_size=self.patch_embed.grid_size if use_rel_pos_bias else None,
                n_prompts=n_prompts, n_tasks=n_tasks,
            )
            for i in range(depth)])

        self.apply(self._init_weights)
        if self.pos_embed is not None:
            trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        # trunc_normal_(self.mask_token, std=.02)
        self.fix_init_weight()

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, CustomLayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}


def _create_beit(variant, pretrained=False, default_cfg=None, **kwargs):
    default_cfg = default_cfg or default_cfgs[variant]
    if kwargs.get('features_only', None):
        raise RuntimeError('features_only not implemented for Beit models.')

    model = build_model_with_cfg(
        CustomBeit, variant, pretrained,
        default_cfg=default_cfg,
        # FIXME an updated filter fn needed to interpolate rel pos emb if fine tuning to diff model sizes
        pretrained_filter_fn=checkpoint_filter_fn,
        **kwargs)
    return model


@register_model
def beit_base_patch16_224(pretrained=False, **kwargs):
    model_kwargs = dict(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
        use_abs_pos_emb=False, use_rel_pos_bias=True, init_values=0.1, **kwargs)
    model = _create_beit('beit_base_patch16_224', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def beit_base_patch16_384(pretrained=False, **kwargs):
    model_kwargs = dict(
        img_size=384, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
        use_abs_pos_emb=False, use_rel_pos_bias=True, init_values=0.1, **kwargs)
    model = _create_beit('beit_base_patch16_384', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def beit_base_patch16_224_in22k(pretrained=False, **kwargs):
    model_kwargs = dict(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
        use_abs_pos_emb=False, use_rel_pos_bias=True, init_values=0.1, **kwargs)
    model = _create_beit('beit_base_patch16_224_in22k', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def beit_large_patch16_224(pretrained=False, **kwargs):
    model_kwargs = dict(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        use_abs_pos_emb=False, use_rel_pos_bias=True, init_values=1e-5,  **kwargs)
    model = _create_beit('beit_large_patch16_224', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def beit_large_patch16_384(pretrained=False, **kwargs):
    model_kwargs = dict(
        img_size=384, patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        use_abs_pos_emb=False, use_rel_pos_bias=True, init_values=1e-5, **kwargs)
    model = _create_beit('beit_large_patch16_384', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def beit_large_patch16_512(pretrained=False, **kwargs):
    model_kwargs = dict(
        img_size=512, patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        use_abs_pos_emb=False, use_rel_pos_bias=True, init_values=1e-5, **kwargs)
    model = _create_beit('beit_large_patch16_512', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def beit_large_patch16_224_in22k(pretrained=False, **kwargs):
    model_kwargs = dict(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        use_abs_pos_emb=False, use_rel_pos_bias=True, init_values=1e-5,  **kwargs)
    model = _create_beit('beit_large_patch16_224_in22k', pretrained=pretrained, **model_kwargs)
    return model


def convert_beit_bitfit(beit, n_tasks):
    for blk in beit.blocks:
        for name, module in blk.named_children():
            if isinstance(module, CustomLayerNorm):
                norm_bitfit = CustomLayerNorm(
                    n_tasks,
                    module.normalized_shape,
                    module.eps,
                    module.elementwise_affine
                ).to(module.bias.device)
                norm_bitfit.bias.data = repeat(module.bias.data, '... -> T ...', T=n_tasks)
                setattr(blk, name, norm_bitfit)
                
            elif isinstance(module, CustomMlp):
                for subname, submodule in module.named_children():
                    if isinstance(submodule, CustomLinear):
                        fc_bitfit = CustomLinear(
                            n_tasks,
                            submodule.in_features,
                            submodule.out_features
                        ).to(submodule.bias.device)
                        fc_bitfit.bias.data = repeat(submodule.bias.data, '... -> T ...', T=n_tasks)
                        setattr(getattr(blk, name), subname, fc_bitfit)