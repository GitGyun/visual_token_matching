import pytorch_lightning as pl
import torch
import torchvision.transforms as T
from einops import rearrange, repeat
import os

from model.model_factory import get_model

from dataset.dataloader_factory import get_train_loader, get_validation_loaders, generate_support_data
from dataset.taskonomy_constants import SEMSEG_CLASSES, TASKS_SEMSEG
from dataset.utils import to_device, mix_fivecrop

from .optim import get_optimizer
from .loss import compute_loss, compute_metric
from .visualize import visualize_batch, postprocess_depth, postprocess_semseg
from .miou_fss import AverageMeter


class LightningTrainWrapper(pl.LightningModule):
    def __init__(self, config, verbose=True, load_pretrained=True):
        '''
        Pytorch lightning wrapper for Visual Token Matching.
        '''
        super().__init__()

        # load model.
        self.model = get_model(config, verbose=verbose, load_pretrained=load_pretrained)
        self.config = config
        self.verbose = verbose

        # tools for validation.
        self.miou_evaluator = AverageMeter(range(len(SEMSEG_CLASSES)))
        self.crop = T.Compose([
            T.FiveCrop(config.img_size),
            T.Lambda(lambda crops: torch.stack([crop for crop in crops]))
        ])
        self.support_data = self.load_support_data()
        
        # save hyper=parameters
        self.save_hyperparameters()

    def load_support_data(self, data_path='support_data.pth'):
        '''
        Load support data for validation.
        '''
        # generate support data if not exists.
        if os.path.exists(data_path):
            support_data = torch.load(data_path)
            print('loaded support data')
        else:
            print('generating support data...')
            support_data = generate_support_data(self.config)
            torch.save(support_data, data_path)
        
        # convert to proper precision
        if self.config.precision == 'fp16':
            support_data = to_device(support_data, dtype=torch.half)
        elif self.config.precision == 'bf16':
            support_data = to_device(support_data, dtype=torch.bfloat16)
        
        return support_data

    def configure_optimizers(self):
        '''
        Prepare optimizer and lr scheduler.
        '''
        optimizer, self.lr_scheduler = get_optimizer(self.config, self.model)
        return optimizer
    
    def train_dataloader(self, verbose=True):
        '''
        Prepare training loader.
        '''
        return get_train_loader(self.config, verbose=(self.verbose and verbose))
    
    def val_dataloader(self, verbose=True):
        '''
        Prepare validation loaders.
        '''
        if not self.config.no_eval:
            val_loaders, loader_tag = get_validation_loaders(self.config, verbose=(self.verbose and verbose))
            self.valid_tasks = list(val_loaders.keys())
            self.valid_tag = loader_tag
            
            return list(val_loaders.values())
        
    def forward(self, *args, **kwargs):
        '''
        Forward data to model.
        '''
        return self.model(*args, **kwargs)
    
    def training_step(self, batch, batch_idx):
        '''
        A single training iteration.
        '''
        # forward model and compute loss.
        loss = compute_loss(self.model, batch, self.config)

        # schedule learning rate.
        self.lr_scheduler.step(self.global_step)
        
        # log losses and learning rate.
        log_dict = {
            'training/loss': loss.detach(),
            'training/lr': self.lr_scheduler.lr,
            'step': self.global_step,
        }
        self.log_dict(
            log_dict,
            logger=True,
            on_step=True,
            sync_dist=True,
        )

        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        '''
        Evaluate few-shot performance on validation dataset.
        '''
        task = self.valid_tasks[dataloader_idx]
        
        # query data
        X, Y, M = batch

        # support data
        X_S, Y_S, M_S, t_idx = to_device(self.support_data[task], X.device)
        t_idx = t_idx.long()
        T = Y_S.size(1)

        # five-crop query images to 224 x 224 and reshape for matching
        X_crop = repeat(self.crop(X), 'F B C H W -> 1 T (F B) C H W', T=T)

        # predict labels on each crop and reshape to five-cropped batches
        if self.config.model == 'TSN':
            Y_pred_crop = self.model(X_crop, t_idx=t_idx, sigmoid=('segment_semantic' not in task))
        else:
            # ignore masked region in support label
            Y_S_in = torch.where(M_S.bool(), Y_S, torch.ones_like(Y_S) * self.config.mask_value)
            Y_pred_crop = self.model(X_S, Y_S_in, X_crop, t_idx=t_idx, sigmoid=('segment_semantic' not in task))
        Y_pred_crop = rearrange(Y_pred_crop, '1 T (F B) 1 H W -> F B T H W', F=5)

        # remix the cropped predictions into a whole prediction
        Y_pred = mix_fivecrop(Y_pred_crop, base_size=X.size(-1), crop_size=X_crop.size(-1))

        # discretization for semantic segmentation
        if 'segment_semantic' in task:
            if self.config.model not in ['HSNet', 'VAT']:
                Y_pred = Y_pred.sigmoid()
            Y_pred = (Y_pred > self.config.semseg_threshold).float()

        # compute evaluation metric
        metric = compute_metric(Y, Y_pred, M, task, self.miou_evaluator)
        metric *= len(X)
        
        # visualize first batch
        if batch_idx == 0:
            vis_batch = (X, Y, M, Y_pred)
            self.vis_images(vis_batch, task)

        return metric, len(X)
        
    def validation_epoch_end(self, validation_step_outputs):
        '''
        Aggregate losses of all validation datasets and log them into tensorboard.
        '''
        avg_loss = []
        log_dict = {'step': self.global_step}

        for task, losses_batch in zip(self.valid_tasks, validation_step_outputs):
            N_total = sum([losses[1] for losses in losses_batch])
            loss_pred = sum([losses[0] for losses in losses_batch]) / N_total

            # log task-specific errors
            if 'segment_semantic' in task:
                if TASKS_SEMSEG.index(task) == 0:
                    loss_pred = 1 - self.miou_evaluator.compute_iou()[0].cpu().item()
                    log_dict[f'{self.valid_tag}/segment_semantic_pred'] = loss_pred
                    avg_loss.append(loss_pred)
            else:
                log_dict[f'{self.valid_tag}/{task}_pred'] = loss_pred
                avg_loss.append(loss_pred)

        # log task-averaged error
        avg_loss = sum(avg_loss) / len(avg_loss)
        if self.global_step > 0:
            log_dict[f'summary/{self.valid_tag}_pred'] = avg_loss

        self.log_dict(
            log_dict,
            logger=True,
            sync_dist=True,
        )
        
        # reset miou evaluator
        self.miou_evaluator = AverageMeter(range(len(SEMSEG_CLASSES)))
        
    @pl.utilities.rank_zero_only
    def vis_images(self, batch, task, vis_shot=-1, **kwargs):
        '''
        Visualize query prediction into tensorboard.
        '''
        X, Y, M, Y_pred = batch

        # choose proper subset.
        if vis_shot > 0:
            X = X[:vis_shot]
            Y = Y[:vis_shot]
            M = M[:vis_shot]
            Y_pred = Y_pred[:vis_shot]
        
        # set task-specific post-processing function for visualization
        if task == 'depth_zbuffer':
            postprocess_fn = postprocess_depth
        elif 'segment_semantic' in task:
            postprocess_fn = postprocess_semseg
        else:
            postprocess_fn = None
        
        # visualize batch
        vis = visualize_batch(X, Y, M, Y_pred, postprocess_fn=postprocess_fn, **kwargs)
        self.logger.experiment.add_image(f'{self.valid_tag}/{task}', vis, self.global_step)