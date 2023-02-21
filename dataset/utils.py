import torch
import torch.nn.functional as F
import torchvision.transforms as T
import numpy as np


def crop_arrays(*arrays, base_size=(256, 256), img_size=(224, 224), random=True, get_offsets=False, offset_cuts=None):
    '''
    Crop arrays from base_size to img_size.
    Apply center crop if not random.
    '''
    if base_size[0] == img_size[0] and base_size[1] == img_size[1]:
        if get_offsets:
            return arrays, (0, 0)
        else:
            return arrays

    if random:
        if offset_cuts is not None:
            off_H = np.random.randint(min(base_size[0] - img_size[0],  offset_cuts[0]))
            off_W = np.random.randint(min(base_size[1] - img_size[1],  offset_cuts[1]))
        else:
            off_H = np.random.randint(base_size[0] - img_size[0])
            off_W = np.random.randint(base_size[1] - img_size[1])
    else:
        if offset_cuts is not None:
            off_H = min(
                max(0, offset_cuts[0] - (base_size[0] - img_size[0]) // 2),
                (base_size[0] - img_size[0]) // 2
            )
            off_W = min(
                max(0, offset_cuts[1] - (base_size[1] - img_size[1]) // 2),
                (base_size[1] - img_size[1]) // 2
            )
        else:
            off_H = (base_size[0] - img_size[0]) // 2
            off_W = (base_size[1] - img_size[1]) // 2

    slice_H = slice(off_H, off_H + img_size[0])
    slice_W = slice(off_W, off_W + img_size[1])

    arrays_cropped = []
    for array in arrays:
        if array is not None:
            assert array.ndim >= 2
            array_cropped = array[..., slice_H, slice_W]
            arrays_cropped.append(array_cropped)
        else:
            arrays_cropped.append(array)

    if get_offsets:
        return arrays_cropped, (off_H, off_W)
    else:
        return arrays_cropped


def mix_fivecrop(x_crop, base_size=256, crop_size=224):
    margin = base_size - crop_size
    submargin = margin // 2
    
    ### Five-pad each crops
    pads = [
        T.Pad((0, 0, margin, margin)),
        T.Pad((margin, 0, 0, margin)),
        T.Pad((0, margin, margin, 0)),
        T.Pad((margin, margin, 0, 0)),
        T.Pad((submargin, submargin, submargin, submargin)),
    ]
    x_pad = []
    for x_, pad in zip(x_crop, pads):
        x_pad.append(pad(x_))
    x_pad = torch.stack(x_pad)

    x_avg = torch.zeros_like(x_pad[0])

    ### Mix padded crops
    # top-left corner
    x_avg[:, :, :submargin, :margin] = x_pad[0][:, :, :submargin, :margin]
    x_avg[:, :, submargin:margin, :submargin] = x_pad[0][:, :, submargin:margin, :submargin]
    x_avg[:, :, submargin:margin, submargin:margin] = (x_pad[0][:, :, submargin:margin, submargin:margin] + \
                                                       x_pad[4][:, :, submargin:margin, submargin:margin]) / 2

    # top-right corner
    x_avg[:, :, :submargin, -margin:] = x_pad[1][:, :, :submargin, -margin:]
    x_avg[:, :, submargin:margin, -submargin:] = x_pad[1][:, :, submargin:margin, -submargin:]
    x_avg[:, :, submargin:margin, -margin:-submargin] = (x_pad[1][:, :, submargin:margin, -margin:-submargin] + \
                                                         x_pad[4][:, :, submargin:margin, -margin:-submargin]) / 2

    # bottom-left corner
    x_avg[:, :, -submargin:, :margin] = x_pad[2][:, :, -submargin:, :margin]
    x_avg[:, :, -margin:-submargin:, :submargin] = x_pad[2][:, :, -margin:-submargin, :submargin]
    x_avg[:, :, -margin:-submargin, submargin:margin] = (x_pad[2][:, :, -margin:-submargin, submargin:margin] + \
                                                         x_pad[4][:, :, -margin:-submargin, submargin:margin]) / 2

    # bottom-left corner
    x_avg[:, :, -submargin:, -margin:] = x_pad[3][:, :, -submargin:, -margin:]
    x_avg[:, :, -margin:-submargin, -submargin:] = x_pad[3][:, :, -margin:-submargin, -submargin:]
    x_avg[:, :, -margin:-submargin, -margin:-submargin] = (x_pad[3][:, :, -margin:-submargin, -margin:-submargin] + \
                                                           x_pad[4][:, :, -margin:-submargin, -margin:-submargin]) / 2

    # top side
    x_avg[:, :, :submargin, margin:-margin] = (x_pad[0][:, :, :submargin, margin:-margin] + \
                                               x_pad[1][:, :, :submargin, margin:-margin]) / 2
    x_avg[:, :, submargin:margin, margin:-margin] = (x_pad[0][:, :, submargin:margin, margin:-margin] + \
                                                     x_pad[1][:, :, submargin:margin, margin:-margin] + \
                                                     x_pad[4][:, :, submargin:margin, margin:-margin]) / 3

    # right side
    x_avg[:, :, margin:-margin, -submargin:] = (x_pad[1][:, :, margin:-margin, -submargin:] + \
                                                x_pad[3][:, :, margin:-margin, -submargin:]) / 2
    x_avg[:, :, margin:-margin, -margin:-submargin] = (x_pad[1][:, :, margin:-margin, -margin:-submargin] + \
                                                       x_pad[3][:, :, margin:-margin, -margin:-submargin] + \
                                                       x_pad[4][:, :, margin:-margin, -margin:-submargin]) / 3

    # bottom side
    x_avg[:, :, -submargin:, margin:-margin] = (x_pad[2][:, :, -submargin:, margin:-margin] + \
                                                x_pad[3][:, :, -submargin:, margin:-margin]) / 2
    x_avg[:, :, -margin:-submargin:, margin:-margin] = (x_pad[2][:, :, -margin:-submargin, margin:-margin] + \
                                                        x_pad[3][:, :, -margin:-submargin, margin:-margin] + \
                                                        x_pad[4][:, :, -margin:-submargin, margin:-margin]) / 3

    # left side
    x_avg[:, :, margin:-margin, :submargin] = (x_pad[0][:, :, margin:-margin, :submargin] + \
                                               x_pad[2][:, :, margin:-margin, :submargin]) / 2
    x_avg[:, :, margin:-margin, submargin:margin] = (x_pad[0][:, :, margin:-margin, submargin:margin] + \
                                                     x_pad[2][:, :, margin:-margin, submargin:margin] + \
                                                     x_pad[4][:, :, margin:-margin, submargin:margin]) / 3

    # center
    x_avg[:, :, margin:-margin, margin:-margin] = (x_pad[0][:, :, margin:-margin, margin:-margin] + \
                                                   x_pad[1][:, :, margin:-margin, margin:-margin] + \
                                                   x_pad[2][:, :, margin:-margin, margin:-margin] + \
                                                   x_pad[3][:, :, margin:-margin, margin:-margin] + \
                                                   x_pad[4][:, :, margin:-margin, margin:-margin]) / 5
    
    return x_avg


def to_device(data, device=None, dtype=None):
    '''
    Load data with arbitrary structure on device.
    '''
    def to_device_wrapper(data):
        if isinstance(data, torch.Tensor):
            return data.to(device=device, dtype=dtype)
        elif isinstance(data, tuple):
            return tuple(map(to_device_wrapper, data))
        elif isinstance(data, list):
            return list(map(to_device_wrapper, data))
        elif isinstance(data, dict):
            return {key: to_device_wrapper(data[key]) for key in data}
        else:
            return data
            
    return to_device_wrapper(data)


def pad_by_reflect(x, padding=1):
    x = torch.cat((x[..., :padding], x, x[..., -padding:]), dim=-1)
    x = torch.cat((x[..., :padding, :], x, x[..., -padding:, :]), dim=-2)
    return x


class SobelEdgeDetector:
    def __init__(self, kernel_size=5, sigma=1):
        self.kernel_size = kernel_size
        self.sigma = sigma

        # compute gaussian kernel
        size = kernel_size // 2
        x, y = np.mgrid[-size:size+1, -size:size+1]
        normal = 1 / (2.0 * np.pi * sigma**2)
        g =  np.exp(-((x**2 + y**2) / (2.0*sigma**2))) * normal

        self.gaussian_kernel = torch.from_numpy(g)[None, None, :, :].float()
        self.Kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float)[None, None, :, :]
        self.Ky = -torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float)[None, None, :, :]

    def detect(self, img, normalize=True):
        squeeze = False
        if len(img.shape) == 3:
            img = img[None, ...]
            squeeze = True

        img = pad_by_reflect(img, padding=self.kernel_size//2)
        img = F.conv2d(img, self.gaussian_kernel.repeat(1, img.size(1), 1, 1))

        img = pad_by_reflect(img, padding=1)
        Gx = F.conv2d(img, self.Kx)
        Gy = F.conv2d(img, self.Ky)

        G = (Gx.pow(2) + Gy.pow(2)).pow(0.5)
        if normalize:
            G = G / G.max()
        if squeeze:
            G = G[0]

        return G
