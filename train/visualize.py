from skimage import color
import numpy as np

import torch
import torch.nn.functional as F
from torchvision.utils import make_grid

from dataset.taskonomy_constants import *
               

def visualize_batch(X=None, Y=None, M=None, Y_preds=None, channels=None, size=None, postprocess_fn=None, **kwargs):
    '''
    Visualize a global batch consists of N-shot images and labels for T channels.
    It is assumed that images are shared by all channels, thus convert channels into RGB and visualize at once.
    '''
    
    vis = []
    
    # shape check
    assert X is not None or Y is not None or Y_preds is not None
    
    # visualize image
    if X is not None:
        img = X.cpu().float()
        vis.append(img)
    else:
        img = None
        
    # flatten labels and masks
    Ys = []
    Ms = []
    if Y is not None:
        Ys.append((Y, None))
        Ms.append(M)
    if Y_preds is not None:
        if isinstance(Y_preds, torch.Tensor):
            Ys.append((Y_preds, Y))
            Ms.append(None)
        elif isinstance(Y_preds, (tuple, list)):
            if Y is not None:
                for Y_pred in Y_preds:
                    Ys.append((Y_pred, Y))
                    Ms.append(None)
            else:
                for Y_pred in Y_preds:
                    Ys.append((Y_pred, None))
                    Ms.append(None)
        else:
            ValueError(f'unsupported predictions type: {type(Y_preds)}')

    # visualize labels
    if len(Ys) > 0:
        for Y, Y_gt in Ys:
            Y = Y.cpu().float()
            if Y_gt is not None:
                Y_gt = Y_gt.cpu().float()

            if channels is None:
                channels = list(range(Y.size(1)))

            label = Y[:, channels].clip(0, 1)
            if Y_gt is not None:
                label_gt = Y_gt[:, channels].clip(0, 1)
            else:
                label_gt = None

            # fill masked region with random noise
            if M is not None:
                assert Y.shape == M.shape
                M = M.cpu().float()
                label = torch.where(M[:, channels].bool(),
                                    label,
                                    torch.rand_like(label))
                if Y_gt is not None:
                    label_gt = Y_gt[:, channels].clip(0, 1)
                    label_gt = torch.where(M[:, channels].bool(),
                                           label_gt,
                                           torch.rand_like(label_gt))

            if postprocess_fn is not None:
                label = postprocess_fn(label, img, label_gt=label_gt)
            
            label = visualize_label_as_rgb(label)
            vis.append(label)

    nrow = len(vis[0])
    vis = torch.cat(vis)
    if size is not None:
        vis = F.interpolate(vis, size)
    vis = make_grid(vis, nrow=nrow, **kwargs)
    vis = vis.float()
    
    return vis


def postprocess_depth(label, img=None, **kwargs):
    label = 0.6*label + 0.4
    label = torch.exp(label * np.log(2.0**16.0)) - 1.0
    label = torch.log(label) / 11.09
    label = (label - 0.64) / 0.18
    label = (label + 1.) / 2
    label = (label*255).byte().float() / 255.
    return label


def postprocess_semseg(label, img=None, **kwargs):
    COLORS = ('red', 'blue', 'yellow', 'magenta', 
              'green', 'indigo', 'darkorange', 'cyan', 'pink', 
              'yellowgreen', 'black', 'darkgreen', 'brown', 'gray',
              'purple', 'darkviolet')
    
    if label.ndim == 4:
        label = label.squeeze(1)
    
    label_vis = []
    if img is not None:
        for img_, label_ in zip(img, label):
            for c in range(len(COLORS)+1):
                label_[0, c] = c

            label_vis.append(torch.from_numpy(color.label2rgb(label_.numpy(),
                                                              image=img_.permute(1, 2, 0).numpy(),
                                                              colors=COLORS,
                                                              kind='overlay')).permute(2, 0, 1))
    else:
        for label_ in label:
            for c in range(len(COLORS)+1):
                label_[0, c] = c

            label_vis.append(torch.from_numpy(color.label2rgb(label_.numpy(),
                                                              colors=COLORS,
                                                              kind='overlay')).permute(2, 0, 1))
    
    label = torch.stack(label_vis)
    
    return label


def visualize_label_as_rgb(label):
    if label.size(1) == 1:
        label = label.repeat(1, 3, 1, 1)
    elif label.size(1) == 2:
        label = torch.cat((label, torch.zeros_like(label[:, :1])), 1)
    elif label.size(1) == 5:
        label = torch.stack((
            label[:, :2].mean(1),
            label[:, 2:4].mean(1),
            label[:, 4]
        ), 1)
    elif label.size(1) != 3:
        assert NotImplementedError
        
    return label