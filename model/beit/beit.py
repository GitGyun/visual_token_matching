import torch
import torch.nn as nn
from einops import rearrange
import numpy as np
from scipy import interpolate

from .beit_factory import create_model as create_custom_model


class BEiTEncoder(nn.Module):
    def __init__(self, model_name='beit_base_patch16_224_in22k',
                 drop_rate=0.0, drop_path_rate=0.1, attn_drop_rate=0.0,
                 n_tasks=0, bitfit=True, n_levels=1):
        super().__init__()
        self.beit = create_custom_model(
            model_name,
            pretrained=False,
            num_classes=0,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
            attn_drop_rate=attn_drop_rate,
            init_scale=0.001,
            n_tasks=(n_tasks if bitfit else 0),
        )
        
        self.model_name = model_name
        self.img_size = self.beit.patch_embed.img_size
        self.grid_size = self.beit.patch_embed.grid_size
        self.patch_size = self.beit.patch_embed.patch_size
        self.embed_dim = self.beit.embed_dim
        self.n_tasks = n_tasks
        self.n_levels = n_levels
        self.feature_blocks = [level * (len(self.beit.blocks) // self.n_levels) - 1 for level in range(1, self.n_levels+1)]

    def bias_parameters(self):
        for key, param in self.beit.named_parameters():
            if key.split('.')[0] == 'blocks' and key.split('.')[-1] == 'bias' and key.split('.')[-3] != 'patch_embed':
                yield param

    def bias_parameter_names(self):
        names = []
        for key, _ in self.beit.named_parameters():
            if key.split('.')[0] == 'blocks' and key.split('.')[-1] == 'bias' and key.split('.')[-3] != 'patch_embed':
                names.append(f'beit.{key}')
        return names
        
    def tokenize(self, x):
        # project image patches to tokens
        x = self.beit.patch_embed(x)
        
        # add CLS token
        x = torch.cat((self.beit.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        
        # positional embedding
        if self.beit.pos_embed is not None:
            x = x + self.beit.pos_embed
        x = self.beit.pos_drop(x)
        
        return x
        
    def forward(self, x, t_idx=None, get_features=False):
        x = self.tokenize(x)
        rel_pos_bias = self.beit.rel_pos_bias() if self.beit.rel_pos_bias is not None else None
        
        if get_features:
            features = []

        # transformer blocks
        for blk_idx in range(len(self.beit.blocks)):
            x = self.beit.blocks[blk_idx](x, rel_pos_bias=rel_pos_bias, t_idx=t_idx)
            
            if get_features and blk_idx in self.feature_blocks:
                feature = x[:, 1:]
                feature = rearrange(feature, 'B (H W) C -> B C H W', H=self.grid_size[0]).contiguous()
                features.append(feature)
        
        if get_features:
            return features
        else:
            # cut off CLS token, then rearrange into spatial maps
            x = x[:, 1:]
            x = rearrange(x, 'B (H W) C -> B C H W', H=self.grid_size[0], W=self.grid_size[1]).contiguous()

            return x
    

def load_state_dict(model, state_dict, prefix='', ignore_missing="relative_position_index"):
    missing_keys = []
    unexpected_keys = []
    error_msgs = []
    # copy state_dict so _load_from_state_dict can modify it
    metadata = getattr(state_dict, '_metadata', None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    def load(module, prefix=''):
        local_metadata = {} if metadata is None else metadata.get(
            prefix[:-1], {})
        module._load_from_state_dict(
            state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + '.')

    load(model, prefix=prefix)

    warn_missing_keys = []
    ignore_missing_keys = []
    for key in missing_keys:
        keep_flag = True
        for ignore_key in ignore_missing.split('|'):
            if ignore_key in key:
                keep_flag = False
                break
        if keep_flag:
            warn_missing_keys.append(key)
        else:
            ignore_missing_keys.append(key)

    missing_keys = warn_missing_keys

    if len(error_msgs) > 0:
        print('\n'.join(error_msgs))


def load_beit_ckpt(model, ckpt_path, n_bitfit_tasks=0, verbose=True):
    model_key = 'model|module'
    checkpoint = torch.load(ckpt_path, map_location='cpu')

    if verbose:
        print("Load ckpt from %s" % ckpt_path)
    checkpoint_model = None
    for model_key in model_key.split('|'):
        if model_key in checkpoint:
            checkpoint_model = checkpoint[model_key]
            if verbose:
                print("Load state_dict by model_key = %s" % model_key)
            break
    if checkpoint_model is None:
        checkpoint_model = checkpoint
    state_dict = model.state_dict()
    for k in ['head.weight', 'head.bias']:
        if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
            if verbose:
                print(f"Removing key {k} from pretrained checkpoint")
            del checkpoint_model[k]

    if model.use_rel_pos_bias and "rel_pos_bias.relative_position_bias_table" in checkpoint_model:
        if verbose:
            print("Expand the shared relative position embedding to each transformer block. ")
        num_layers = model.get_num_layers()
        rel_pos_bias = checkpoint_model["rel_pos_bias.relative_position_bias_table"]
        for i in range(num_layers):
            checkpoint_model["blocks.%d.attn.relative_position_bias_table" % i] = rel_pos_bias.clone()

        checkpoint_model.pop("rel_pos_bias.relative_position_bias_table")

    all_keys = list(checkpoint_model.keys())
    for key in all_keys:
        if "relative_position_index" in key:
            checkpoint_model.pop(key)

        if "relative_position_bias_table" in key:
            rel_pos_bias = checkpoint_model[key]
            src_num_pos, num_attn_heads = rel_pos_bias.size()
            dst_num_pos, _ = model.state_dict()[key].size()
            try:
                dst_patch_shape = model.patch_embed.patch_size
            except AttributeError:
                dst_patch_shape = model.patch_embed[0].patch_size
            if dst_patch_shape[0] != dst_patch_shape[1]:
                raise NotImplementedError()
            num_extra_tokens = dst_num_pos - (dst_patch_shape[0] * 2 - 1) * (dst_patch_shape[1] * 2 - 1)
            src_size = int((src_num_pos - num_extra_tokens) ** 0.5)
            dst_size = int((dst_num_pos - num_extra_tokens) ** 0.5)
            if src_size != dst_size:
                if verbose:
                    print("Position interpolate for %s from %dx%d to %dx%d" % (
                    key, src_size, src_size, dst_size, dst_size))
                extra_tokens = rel_pos_bias[-num_extra_tokens:, :]
                rel_pos_bias = rel_pos_bias[:-num_extra_tokens, :]

                def geometric_progression(a, r, n):
                    return a * (1.0 - r ** n) / (1.0 - r)

                left, right = 1.01, 1.5
                while right - left > 1e-6:
                    q = (left + right) / 2.0
                    gp = geometric_progression(1, q, src_size // 2)
                    if gp > dst_size // 2:
                        right = q
                    else:
                        left = q

                # if q > 1.090307:
                #     q = 1.090307

                dis = []
                cur = 1
                for i in range(src_size // 2):
                    dis.append(cur)
                    cur += q ** (i + 1)

                r_ids = [-_ for _ in reversed(dis)]

                x = r_ids + [0] + dis
                y = r_ids + [0] + dis

                t = dst_size // 2.0
                dx = np.arange(-t, t + 0.1, 1.0)
                dy = np.arange(-t, t + 0.1, 1.0)

                if verbose:
                    print("Original positions = %s" % str(x))
                    print("Target positions = %s" % str(dx))

                all_rel_pos_bias = []

                for i in range(num_attn_heads):
                    z = rel_pos_bias[:, i].view(src_size, src_size).float().numpy()
                    f = interpolate.interp2d(x, y, z, kind='cubic')
                    all_rel_pos_bias.append(
                        torch.Tensor(f(dx, dy)).contiguous().view(-1, 1).to(rel_pos_bias.device))

                rel_pos_bias = torch.cat(all_rel_pos_bias, dim=-1)

                new_rel_pos_bias = torch.cat((rel_pos_bias, extra_tokens), dim=0)
                checkpoint_model[key] = new_rel_pos_bias

    # interpolate position embedding
    if 'pos_embed' in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model['pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.patch_embed.num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int(num_patches ** 0.5)
        # class_token and dist_token are kept unchanged
        if orig_size != new_size:
            if verbose:
                print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            checkpoint_model['pos_embed'] = new_pos_embed
            

    if n_bitfit_tasks > 0:
        for key in checkpoint_model:
            if key.split('.')[0] == 'blocks' and key.split('.')[-1] == 'bias' and key.split('.')[-3] != 'patch_embed':
                checkpoint_model[key] = torch.stack([checkpoint_model[key] for _ in range(n_bitfit_tasks)]).contiguous()
                
    load_state_dict(model, checkpoint_model)
