import torch
import torch.nn.functional as F
from einops import rearrange
import math

from dataset.taskonomy_constants import TASKS, TASKS_SEMSEG, SEMSEG_CLASSES
from .miou_fss import Evaluator


SEMSEG_IDXS = [TASKS.index(task) for task in TASKS_SEMSEG]


def generate_semseg_mask(t_idx):
    '''
    Generate binary mask whether the task is semantic segmentation (1) or not (0).
    '''
    semseg_mask = torch.zeros_like(t_idx, dtype=bool)
    for semseg_idx in SEMSEG_IDXS:
        semseg_mask = torch.logical_or(semseg_mask, t_idx == semseg_idx)

    return semseg_mask


def hybrid_loss(Y_src, Y_tgt, M, t_idx):
    '''
    Compute l1 loss for continuous tasks and bce loss for semantic segmentation.
    [loss_args]
        Y_src: unnormalized prediction of shape (B, T, N, 1, H, W)
        Y_tgt: normalized GT of shape (B, T, N, 1, H, W)
        M    : mask for loss computation of shape (B, T, N, 1, H, W)
        t_idx: task index of shape (B, T)
    '''
    # prediction loss
    loss_seg = F.binary_cross_entropy_with_logits(Y_src, Y_tgt, reduction='none')
    loss_con = F.l1_loss(Y_src.sigmoid(), Y_tgt, reduction='none')

    # loss masking
    loss_seg = rearrange((M * loss_seg), 'B T ... -> (B T) ...')
    loss_con = rearrange((M * loss_con), 'B T ... -> (B T) ...')
    t_idx = rearrange(t_idx, 'B T -> (B T)')

    # loss switching
    semseg_mask = generate_semseg_mask(t_idx)
    semseg_mask = rearrange(semseg_mask, 'B -> B 1 1 1 1').float()
    loss = (semseg_mask * loss_seg + (1 - semseg_mask) * loss_con).mean()
    
    return loss


def compute_loss(model, train_data, config):
    '''
    Compute episodic training loss for VTM.
    [train_data]
        X    : input image of shape (B, T, N, 3, H, W)
        Y    : output label of shape (B, T, N, 1, H, W)
        M    : output mask of shape (B, T, N, 1, H, W)
        t_idx: task index of shape (B, T)
    '''
    X, Y, M, t_idx = train_data

    # split the batches into support and query
    X_S, X_Q = X.split(math.ceil(X.size(2) / 2), dim=2)
    Y_S, Y_Q = Y.split(math.ceil(Y.size(2) / 2), dim=2)
    M_S, M_Q = M.split(math.ceil(M.size(2) / 2), dim=2)

    # ignore masked region in support label
    Y_S_in = torch.where(M_S.bool(), Y_S, torch.ones_like(Y_S) * config.mask_value)

    # compute loss for query images
    Y_Q_pred = model(X_S, Y_S_in, X_Q, t_idx=t_idx, sigmoid=False)
    loss = hybrid_loss(Y_Q_pred, Y_Q, M_Q, t_idx)
    
    return loss


def normalize_tensor(input_tensor, dim):
    '''
    Normalize Euclidean vector.
    '''
    norm = torch.norm(input_tensor, p='fro', dim=dim, keepdim=True)
    zero_mask = (norm == 0)
    norm[zero_mask] = 1
    out = input_tensor.div(norm)
    out[zero_mask.expand_as(out)] = 0
    return out


def compute_metric(Y, Y_pred, M, task, miou_evaluator=None, stage=0):
    '''
    Compute evaluation metric for each task.
    '''
    # Mean Angle Error
    if task == 'normal':
        pred = normalize_tensor(Y_pred, dim=1)
        gt = normalize_tensor(Y, dim=1)
        deg_diff = torch.rad2deg(2 * torch.atan2(torch.norm(pred - gt, dim=1), torch.norm(pred + gt, dim=1)))
        metric = (M[:, 0] * deg_diff).mean()
        
    # Mean IoU
    elif 'segment_semantic' in task:
        assert miou_evaluator is not None

        area_inter, area_union = Evaluator.classify_prediction(Y_pred.clone().float(), Y.float())
        if stage == 0:
            assert 'segment_semantic' in task
            semseg_class = int(task.split('_')[-1])
            class_id = torch.tensor([SEMSEG_CLASSES.index(semseg_class)]*len(Y_pred), device=Y.device)
        else:
            class_id = torch.tensor([0]*len(Y_pred), device=Y.device)

        area_inter = area_inter.to(Y.device)
        area_union = area_union.to(Y.device)
        miou_evaluator.update(area_inter, area_union, class_id)
        
        metric = 0

    # Mean Squared Error
    else:
        metric = (M * F.mse_loss(Y, Y_pred, reduction='none').pow(0.5)).mean()
        
    return metric