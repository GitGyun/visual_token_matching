import pytorch_lightning as pl
import torch
import torchvision.transforms as T
from einops import rearrange, repeat, reduce
import os

from model.model_factory import get_model

from dataset.dataloader_factory import get_train_loader, get_validation_loaders, generate_support_data, get_eval_loader, base_sizes
from dataset.taskonomy_constants import SEMSEG_CLASSES, TASKS_SEMSEG
from dataset.utils import to_device, mix_fivecrop, crop_arrays

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

        if self.config.stage == 1:
            for attn in self.model.matching_module.matching:
                attn.attn_dropout.p = self.config.attn_dropout
        
        # save hyper=parameters
        self.save_hyperparameters()

    def load_support_data(self, data_path='support_data.pth'):
        '''
        Load support data for validation.
        '''
        if self.config.stage == 0:
            # generate support data if not exists.
            support_data = generate_support_data(self.config, data_path=data_path, verbose=self.verbose)
        else:
            task = f'{self.config.task}_{self.config.channel_idx}' if self.config.task == 'segment_semantic' else self.config.task
            support_data = {task: get_train_loader(self.config, verbose=False, get_support_data=True)}

        if self.verbose:
            print('loaded support data')
        
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
            # use external data from validation split
            if self.config.stage == 0:
                val_loaders, loader_tag = get_validation_loaders(self.config, verbose=(self.verbose and verbose))
                self.valid_tasks = list(val_loaders.keys())
                self.valid_tag = loader_tag
                
                return list(val_loaders.values())
            
            # use second half of support data as validation query
            else:
                assert self.config.shot > 1
                class SubQueryDataset:
                    def __init__(self, data):
                        self.data = data
                        self.n_query = self.data[0].shape[2] // 2
                    
                    def __len__(self):
                        return self.n_query
                    
                    def __getitem__(self, idx):
                        return (self.data[0][0, 0, self.n_query+idx],
                                self.data[1][0, :, self.n_query+idx, 0],
                                self.data[2][0, :, self.n_query+idx, 0])
                    
                valid_task = list(self.support_data.keys())[0]
                dset = SubQueryDataset(self.support_data[valid_task][:3])
                self.valid_tasks = [valid_task]
                self.valid_tag = 'mtest_support'
                    
                return torch.utils.data.DataLoader(dset, shuffle=False, batch_size=len(dset))
                
    def test_dataloader(self, verbose=True):
        '''
        Prepare test loaders.
        '''
        test_loader = get_eval_loader(self.config, self.config.task, split=self.config.test_split,
                                      channel_idx=self.config.channel_idx, verbose=(self.verbose and verbose))
        
        return test_loader
        
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

        if self.config.stage == 0:
            tag = ''
        elif self.config.stage == 1:
            if self.config.task == 'segment_semantic':
                tag = f'_segment_semantic_{self.config.channel_idx}'
            else:
                tag = f'_{self.config.task}'
        
        # log losses and learning rate.
        log_dict = {
            f'training/loss{tag}': loss.detach(),
            f'training/lr{tag}': self.lr_scheduler.lr,
            'step': self.global_step,
        }
        self.log_dict(
            log_dict,
            logger=True,
            on_step=True,
            sync_dist=True,
        )

        return loss
    
    @torch.autocast(device_type='cuda', dtype=torch.float32)
    def inference(self, X, task):
        # support data
        X_S, Y_S, M_S, t_idx = to_device(self.support_data[task], X.device)

        # use first half of support data as validation support
        if self.config.stage == 1:
            n_support = X_S.shape[2] // 2
            base_size = base_sizes[self.config.img_size]
            img_size = (self.config.img_size, self.config.img_size)
            X_S, Y_S, M_S = crop_arrays(X_S[:, :, :n_support],
                                        Y_S[:, :, :n_support],
                                        M_S[:, :, :n_support],
                                        base_size=base_size,
                                        img_size=img_size,
                                        random=False)

        t_idx = t_idx.long()
        T = Y_S.size(1)

        # five-crop query images to 224 x 224 and reshape for matching
        X_crop = repeat(self.crop(X), 'F B C H W -> 1 T (F B) C H W', T=T)

        # predict labels on each crop
        Y_S_in = torch.where(M_S.bool(), Y_S, torch.ones_like(Y_S) * self.config.mask_value)
        Y_pred_crop = self.model(X_S, Y_S_in, X_crop, t_idx=t_idx, sigmoid=('segment_semantic' not in task))

        # remix the cropped predictions into a whole prediction
        Y_pred_crop = rearrange(Y_pred_crop, '1 T (F B) 1 H W -> F B T H W', F=5)
        Y_pred = mix_fivecrop(Y_pred_crop, base_size=X.size(-1), crop_size=X_crop.size(-1))

        return Y_pred
    
    def on_validation_start(self) -> None:
        if self.config.stage == 0:
            self.miou_evaluator = AverageMeter(range(len(SEMSEG_CLASSES)), device=self.device)
        else:
            self.miou_evaluator = AverageMeter(0, device=self.device)
        return super().on_validation_start()
    
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        '''
        Evaluate few-shot performance on validation dataset.
        '''
        task = self.valid_tasks[dataloader_idx]
        
        # get query data
        X, Y, M = batch

        # few-shot inference based on support data
        Y_pred = self.inference(X, task)

        # discretization for semantic segmentation
        if 'segment_semantic' in task:
            Y_pred = (Y_pred.sigmoid() > self.config.semseg_threshold).float()

        # compute evaluation metric
        metric = compute_metric(Y, Y_pred, M, task, self.miou_evaluator, self.config.stage)
        metric *= len(X)
        
        # visualize first batch
        if batch_idx == 0:
            X_vis = rearrange(self.all_gather(X), 'G B ... -> (B G) ...')
            Y_vis = rearrange(self.all_gather(Y), 'G B ... -> (B G) ...')
            M_vis = rearrange(self.all_gather(M), 'G B ... -> (B G) ...')
            Y_pred_vis = rearrange(self.all_gather(Y_pred), 'G B ... -> (B G) ...')
            vis_batch = (X_vis, Y_vis, M_vis, Y_pred_vis)
            self.vis_images(vis_batch, task)

        return metric, torch.tensor(len(X), device=self.device)
        
    def validation_epoch_end(self, validation_step_outputs):
        '''
        Aggregate losses of all validation datasets and log them into tensorboard.
        '''
        if len(self.valid_tasks) == 1:
            validation_step_outputs = (validation_step_outputs,)
        avg_loss = []
        log_dict = {'step': self.global_step}

        for task, losses_batch in zip(self.valid_tasks, validation_step_outputs):
            N_total = sum([losses[1] for losses in losses_batch])
            loss_pred = sum([losses[0] for losses in losses_batch])
            N_total = self.all_gather(N_total).sum()
            loss_pred = self.all_gather(loss_pred).sum()

            loss_pred = loss_pred / N_total

            # log task-specific errors
            if 'segment_semantic' in task:
                if self.config.stage > 0 or TASKS_SEMSEG.index(task) == 0:
                    self.miou_evaluator.intersection_buf = reduce(self.all_gather(self.miou_evaluator.intersection_buf),
                                                                    'G ... -> ...', 'sum')
                    self.miou_evaluator.union_buf = reduce(self.all_gather(self.miou_evaluator.union_buf),
                                                            'G ... -> ...', 'sum')

                    loss_pred = 1 - self.miou_evaluator.compute_iou()[0]

                    if self.config.stage == 0:
                        tag = f'{self.valid_tag}/segment_semantic_pred'
                    else:
                        tag = f'{self.valid_tag}/segment_semantic_{self.config.channel_idx}_pred'

                    log_dict[tag] = loss_pred
                    avg_loss.append(loss_pred)
            else:
                log_dict[f'{self.valid_tag}/{task}_pred'] = loss_pred
                avg_loss.append(loss_pred)

        # log task-averaged error
        if self.config.stage == 0:
            avg_loss = sum(avg_loss) / len(avg_loss)
            log_dict[f'summary/{self.valid_tag}_pred'] = avg_loss

        self.log_dict(
            log_dict,
            logger=True,
            rank_zero_only=True
        )

    def on_test_start(self) -> None:
        if self.config.stage == 0:
            self.miou_evaluator = AverageMeter(range(len(SEMSEG_CLASSES)), device=self.device)
        else:
            self.miou_evaluator = AverageMeter(0, device=self.device)
        return super().on_test_start()
    
    def test_step(self, batch, batch_idx):
        '''
        Evaluate few-shot performance on test dataset.
        '''
        if self.config.task == 'segment_semantic':
            task = f'segment_semantic_{self.config.channel_idx}'
        else:
            task = self.config.task

        # query data
        X, Y, M = batch

        # support data
        Y_pred = self.inference(X, task)

        # discretization for semantic segmentation
        if 'segment_semantic' in task:
            Y_pred = (Y_pred.sigmoid() > self.config.semseg_threshold).float()

        # compute evaluation metric
        metric = compute_metric(Y, Y_pred, M, task, self.miou_evaluator, self.config.stage)
        metric *= len(X)

        return metric, torch.tensor(len(X), device=self.device)
    
    def test_epoch_end(self, test_step_outputs):
        # append test split to save_postfix
        log_name = f'result{self.config.save_postfix}_split:{self.config.test_split}.pth'
        log_path = os.path.join(self.config.result_dir, log_name)
        
        if self.config.task == 'segment_semantic':
            torch.save(self.miou_evaluator, log_path)
        else:
            N_total = sum([losses[1] for losses in test_step_outputs])
            metric = sum([losses[0] for losses in test_step_outputs]) / N_total
            metric = metric.cpu().item()
            torch.save(metric, log_path)
        
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