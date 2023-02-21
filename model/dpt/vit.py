import torch
import torch.nn as nn
import timm
import types
import math
import torch.nn.functional as F


activations = {}


def get_activation(name):
    def hook(model, input, output):
        activations[name] = output

    return hook


def _resize_pos_embed(self, posemb, gs_h, gs_w):
    posemb_tok, posemb_grid = (
        posemb[:, : self.start_index],
        posemb[0, self.start_index :],
    )

    gs_old = int(math.sqrt(len(posemb_grid)))

    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(posemb_grid, size=(gs_h, gs_w), mode="bilinear")
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, gs_h * gs_w, -1)

    posemb = torch.cat([posemb_tok, posemb_grid], dim=1)

    return posemb


def forward_flex(self, x, t_idx=None):
    b, c, h, w = x.shape

    if self.pos_embed is not None:
        pos_embed = self._resize_pos_embed(
            self.pos_embed, h // self.patch_size[1], w // self.patch_size[0]
        )
    else:
        pos_embed = None

    B = x.shape[0]

    if hasattr(self.patch_embed, "backbone"):
        x = self.patch_embed.backbone(x)
        if isinstance(x, (list, tuple)):
            x = x[-1]  # last feature if backbone outputs list/tuple of features

    if isinstance(self.patch_embed, nn.ModuleList):
        assert t_idx is not None
        x = torch.cat([self.patch_embed[t_idx_].proj(x_[None]) for t_idx_, x_ in zip(t_idx, x)])
    else:
        x = self.patch_embed.proj(x)
    x = x.flatten(2).transpose(1, 2)

    if getattr(self, "dist_token", None) is not None:
        cls_tokens = self.cls_token.expand(
            B, -1, -1
        )  # stole cls_tokens impl from Phil Wang, thanks
        dist_token = self.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim=1)
    else:
        cls_tokens = self.cls_token.expand(
            B, -1, -1
        )  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)

    if pos_embed is not None:
        x = x + pos_embed
    x = self.pos_drop(x)

    for blk in self.blocks:
        x = blk(x)

    x = self.norm(x)

    return x


def _make_vit_backbone(
    model,
    features=[96, 192, 384, 768],
    hooks=[2, 5, 8, 11],
    vit_features=768,
    start_index=1,
):
    pretrained = nn.Module()

    pretrained.model = model
    pretrained.model.blocks[hooks[0]].register_forward_hook(get_activation("1"))
    pretrained.model.blocks[hooks[1]].register_forward_hook(get_activation("2"))
    pretrained.model.blocks[hooks[2]].register_forward_hook(get_activation("3"))
    pretrained.model.blocks[hooks[3]].register_forward_hook(get_activation("4"))

    pretrained.activations = activations

    # 32, 48, 136, 384
    pretrained.act_postprocess1 = nn.Sequential(
        nn.Conv2d(
            in_channels=vit_features,
            out_channels=features[0],
            kernel_size=1,
            stride=1,
            padding=0,
        ),
        nn.ConvTranspose2d(
            in_channels=features[0],
            out_channels=features[0],
            kernel_size=4,
            stride=4,
            padding=0,
            bias=True,
            dilation=1,
            groups=1,
        ),
    )

    pretrained.act_postprocess2 = nn.Sequential(
        nn.Conv2d(
            in_channels=vit_features,
            out_channels=features[1],
            kernel_size=1,
            stride=1,
            padding=0,
        ),
        nn.ConvTranspose2d(
            in_channels=features[1],
            out_channels=features[1],
            kernel_size=2,
            stride=2,
            padding=0,
            bias=True,
            dilation=1,
            groups=1,
        ),
    )

    pretrained.act_postprocess3 = nn.Sequential(
        nn.Conv2d(
            in_channels=vit_features,
            out_channels=features[2],
            kernel_size=1,
            stride=1,
            padding=0,
        ),
    )

    pretrained.act_postprocess4 = nn.Sequential(
        nn.Conv2d(
            in_channels=vit_features,
            out_channels=features[3],
            kernel_size=1,
            stride=1,
            padding=0,
        ),
        nn.Conv2d(
            in_channels=features[3],
            out_channels=features[3],
            kernel_size=3,
            stride=2,
            padding=1,
        ),
    )

    pretrained.model.start_index = start_index
    pretrained.model.patch_size = [16, 16]

    # We inject this function into the VisionTransformer instances so that
    # we can use it with interpolated position embeddings without modifying the library source.
    pretrained.model.forward_flex = types.MethodType(forward_flex, pretrained.model)
    pretrained.model._resize_pos_embed = types.MethodType(
        _resize_pos_embed, pretrained.model
    )

    return pretrained


def _make_pretrained_vitl16_384(pretrained, in_chans=3):
    model = timm.create_model("vit_large_patch16_384", pretrained=pretrained, in_chans=in_chans)

    hooks = [5, 11, 17, 23]
    return _make_vit_backbone(
        model,
        features=[256, 512, 1024, 1024],
        hooks=hooks,
        vit_features=1024,
    )


def _make_pretrained_vitb16_384(pretrained, in_chans=3):
    model = timm.create_model("vit_base_patch16_384", pretrained=pretrained, in_chans=in_chans)

    hooks = [2, 5, 8, 11]
    return _make_vit_backbone(
        model,
        features=[96, 192, 384, 768],
        hooks=hooks,
    )


def _make_pretrained_vitb16_224(pretrained, in_chans=3):
    model = timm.create_model("vit_base_patch16_224", pretrained=pretrained, in_chans=in_chans)

    hooks = [2, 5, 8, 11]
    return _make_vit_backbone(
        model,
        features=[96, 192, 384, 768],
        hooks=hooks,
    )