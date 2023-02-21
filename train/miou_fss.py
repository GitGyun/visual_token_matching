import torch
from dataset.taskonomy_constants import SEMSEG_CLASSES


class AverageMeter:
    r""" Stores loss, evaluation results """
    def __init__(self, class_ids_interest, device=None):
        if device is None:
            device = torch.device('cpu')
        if isinstance(class_ids_interest, int):
            self.class_ids_interest = torch.tensor([class_ids_interest], device=device)
        else:
            self.class_ids_interest = torch.tensor(class_ids_interest, device=device)

        self.nclass = len(SEMSEG_CLASSES)

        self.intersection_buf = torch.zeros([2, self.nclass], device=device).float()
        self.union_buf = torch.zeros([2, self.nclass], device=device).float()
        self.ones = torch.ones_like(self.union_buf)
        self.loss_buf = []

    def update(self, inter_b, union_b, class_id):
        self.intersection_buf.index_add_(1, class_id, inter_b.float())
        self.union_buf.index_add_(1, class_id, union_b.float())
        
    def class_iou(self, class_id):
        iou = self.intersection_buf.float() / \
              torch.max(torch.stack([self.union_buf, self.ones]), dim=0)[0]
        iou = iou.index_select(1, torch.tensor([class_id], device=iou.device))
        miou = iou[1].mean()
        
        return miou

    def compute_iou(self):
        iou = self.intersection_buf.float() / \
              torch.max(torch.stack([self.union_buf, self.ones]), dim=0)[0]
        iou = iou.index_select(1, self.class_ids_interest)
        miou = iou[1].mean()

        fb_iou = (self.intersection_buf.index_select(1, self.class_ids_interest).sum(dim=1) /
                  self.union_buf.index_select(1, self.class_ids_interest).sum(dim=1)).mean()

        return miou, fb_iou
            

class Evaluator:
    r""" Computes intersection and union between prediction and ground-truth """
    @classmethod
    def initialize(cls):
        pass

    @classmethod
    def classify_prediction(cls, pred_mask, gt_mask):
        # compute intersection and union of each episode in a batch
        area_inter, area_pred, area_gt = [],  [], []
        for _pred_mask, _gt_mask in zip(pred_mask, gt_mask):
            _inter = _pred_mask[_pred_mask == _gt_mask]
            if _inter.size(0) == 0:  # as torch.histc returns error if it gets empty tensor (pytorch 1.5.1)
                _area_inter = torch.tensor([0, 0], device=_pred_mask.device)
            else:
                _area_inter = torch.histc(_inter, bins=2, min=0, max=1)
            area_inter.append(_area_inter)
            area_pred.append(torch.histc(_pred_mask, bins=2, min=0, max=1))
            area_gt.append(torch.histc(_gt_mask, bins=2, min=0, max=1))
        area_inter = torch.stack(area_inter).t()
        area_pred = torch.stack(area_pred).t()
        area_gt = torch.stack(area_gt).t()
        area_union = area_pred + area_gt - area_inter

        return area_inter, area_union
