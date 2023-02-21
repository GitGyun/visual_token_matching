import torch.nn as nn
from einops import rearrange
from .dpt_blocks import _make_fusion_block, Interpolate, _make_encoder


backbone_dict = {
    'vit_base_patch16_224': 'vitb16_224',
    'vit_base_patch16_384': 'vitb16_384',
    'vit_large_patch16_384': 'vitl16_384',
}


class DPT(nn.Module):
    def __init__(self,
            model_name='vit_base_patch16_224',
            features=256,
            use_bn=False,
            pretrained=True,
            in_chans=1,
            out_chans=1
        ):
        super().__init__()
        # Instantiate backbone and reassemble blocks
        self.pretrained, self.scratch = _make_encoder(
            backbone_dict[model_name],
            features,
            pretrained,  # Set to true of you want to train from scratch, uses ImageNet weights
            groups=1,
            expand=False,
            in_chans=in_chans,
        )

        self.scratch.refinenet1 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet2 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet3 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet4 = _make_fusion_block(features, use_bn)

        self.scratch.output_conv = nn.Sequential(
            nn.Conv2d(features, features // 2, kernel_size=3, stride=1, padding=1),
            Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(features // 2, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, out_chans, kernel_size=1, stride=1, padding=0),
        )

        size = [int(model_name.split('_')[-1])]*2
        self.embed_dim = self.pretrained.model.embed_dim
        self.patch_size = (size[0] // 16, size[1] // 16)
        self.n_levels = 4
        self.feature_blocks = [level * (len(self.pretrained.model.blocks) // self.n_levels) - 1 for level in range(1, self.n_levels+1)]
        
    def pretrained_parameters(self):
        return self.pretrained.parameters()
    
    def scratch_parameters(self):
        return self.scratch.parameters()
    
    def encode(self, x, t_idx=None):
        glob = self.pretrained.model.forward_flex(x, t_idx=t_idx)

        layer_1 = rearrange(self.pretrained.activations["1"][:, 1:], 'B (H W) C -> B C H W', H=self.patch_size[0])
        layer_2 = rearrange(self.pretrained.activations["2"][:, 1:], 'B (H W) C -> B C H W', H=self.patch_size[0])
        layer_3 = rearrange(self.pretrained.activations["3"][:, 1:], 'B (H W) C -> B C H W', H=self.patch_size[0])
        layer_4 = rearrange(self.pretrained.activations["4"][:, 1:], 'B (H W) C -> B C H W', H=self.patch_size[0])
        
        return layer_1, layer_2, layer_3, layer_4
    
    def decode(self, features, t_idx=None):
        layer_1, layer_2, layer_3, layer_4 = features
        
        layer_1 = self.pretrained.act_postprocess1(layer_1)
        layer_2 = self.pretrained.act_postprocess2(layer_2)
        layer_3 = self.pretrained.act_postprocess3(layer_3)
        layer_4 = self.pretrained.act_postprocess4(layer_4)

        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        path_4 = self.scratch.refinenet4(layer_4_rn)
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn)
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn)
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)

        out = self.scratch.output_conv(path_1)

        return out

    def forward(self, x, t_idx=None):
        x = self.encode(x, t_idx=t_idx)
        x = self.decode(x, t_idx=t_idx)
        
        return x