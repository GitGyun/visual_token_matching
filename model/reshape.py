import torch
from einops import rearrange, repeat


def get_reshaper(pattern):
    def reshaper(x, contiguous=False, **kwargs):
        if isinstance(x, torch.Tensor):
            x = rearrange(x, pattern, **kwargs)
            if contiguous:
                x = x.contiguous()
            return x
        elif isinstance(x, dict):
            return {key: reshaper(x[key], contiguous=contiguous, **kwargs) for key in x}
        elif isinstance(x, tuple):
            return tuple(reshaper(x_, contiguous=contiguous, **kwargs) for x_ in x)
        elif isinstance(x, list):
            return [reshaper(x_, contiguous=contiguous, **kwargs) for x_ in x]
        else:
            return x
    
    return reshaper


from_6d_to_4d = get_reshaper('B T N C H W -> (B T N) C H W')
from_4d_to_6d = get_reshaper('(B T N) C H W -> B T N C H W')

from_6d_to_3d = get_reshaper('B T N C H W -> (B T) (N H W) C')
from_3d_to_6d = get_reshaper('(B T) (N H W) C -> B T N C H W')


def parse_BTN(x):
    if isinstance(x, torch.Tensor):
        B, T, N = x.size()[:3]
    elif isinstance(x, (tuple, list)):
        B, T, N = x[0].size()[:3]
    elif isinstance(x, dict):
        B, T, N = x[list(x.keys())[0]].size()[:3]
    else:
        raise ValueError(f'unsupported type: {type(x)}')

    return B, T, N


def forward_6d_as_4d(func, x, t_idx=None, **kwargs):
    B, T, N = parse_BTN(x)
        
    x = from_6d_to_4d(x, contiguous=True)
    
    if t_idx is not None:
        t_idx = repeat(t_idx, 'B T -> (B T N)', N=N)
        x = func(x, t_idx=t_idx, **kwargs)
    else:
        x = func(x, **kwargs)
    
    x = from_4d_to_6d(x, B=B, T=T)

    return x
