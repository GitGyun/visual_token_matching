import pytorch_lightning as pl
from pytorch_lightning.callbacks.progress.tqdm_progress import TQDMProgressBar
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

import os
import sys
import warnings
import shutil
import random
import tqdm

import numpy as np
import torch


def configure_experiment(config, **kwargs):
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=pl.utilities.warnings.PossibleUserWarning)
    
    # set seeds
    set_seeds(config.seed)
    
    # set directories
    log_dir, save_dir = set_directories(config, **kwargs)

    # create lightning callbacks
    callbacks = set_callbacks(config, save_dir, "summary/mtrain_valid_pred")

    # create tensorboard logger
    logger = CustomTBLogger(log_dir, name='', version='', default_hp_metric=False)
    
    # create profiler
    profiler = pl.profilers.PyTorchProfiler(log_dir) if config.profile_mode else None
        
    # parse precision
    precision = int(config.precision.strip('fp')) if config.precision in ['fp16', 'fp32'] else config.precision
        
    # choose accelerator
    strategy = set_strategy(config.strategy)
    
    return logger, save_dir, callbacks, profiler, precision, strategy


def set_seeds(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def set_directories(config, root_dir='experiments', exp_name='', log_dir='logs', save_dir='checkpoints',
                    create_log_dir=True, create_save_dir=True, dir_postfix='', finetune_name=''):
    # make an experiment name
    if exp_name == '':
        if config.task == '':
            exp_name = config.exp_name = f'{config.model}_fold:{config.task_fold}{config.name_postfix}'
        else:
            exp_name = config.exp_name = f'{config.model}_task:{config.task}{config.name_postfix}'
    
    # create the root directory
    os.makedirs(root_dir, exist_ok=True)

    # set logging directory
    if create_log_dir:
        os.makedirs(os.path.join(root_dir, config.log_dir), exist_ok=True)
        log_root = os.path.join(root_dir, config.log_dir, exp_name + dir_postfix)
        os.makedirs(log_root, exist_ok=True)
        if finetune_name != '':
            log_root = os.path.join(log_root, finetune_name)
            os.makedirs(log_root, exist_ok=True)
        log_dir = os.path.join(log_root, log_dir)

        # reset the logging directory if exists
        if os.path.exists(log_dir) and not (config.continue_mode or config.skip_mode):
            shutil.rmtree(log_dir)
        os.makedirs(log_dir, exist_ok=True)
    else:
        log_dir = None

    # set saving directory
    if create_save_dir:
        save_root = os.path.join(root_dir, config.save_dir, exp_name + dir_postfix)
        if finetune_name != '':
            save_root = os.path.join(save_root, finetune_name)
        save_dir = os.path.join(save_root, save_dir)

        # create the saving directory if checkpoint doesn't exist or in skipping mode,
        # otherwise ask user to reset it
        if os.path.exists(save_dir) and int(os.environ.get('LOCAL_RANK', 0)) == 0:
            if config.continue_mode:
                print(f'resume from checkpoint ({exp_name})')
            elif config.skip_mode:
                print(f'skip the existing checkpoint ({exp_name})')
                sys.exit()
            elif config.debug_mode or config.reset_mode:
                print(f'remove existing checkpoint ({exp_name})')
                shutil.rmtree(save_dir)
            else:
                while True:
                    print(f'redundant experiment name! ({exp_name}) remove existing checkpoints? (y/n)')
                    inp = input()
                    if inp == 'y':
                        shutil.rmtree(save_dir)
                        break
                    elif inp == 'n':
                        print('quit')
                        sys.exit()
                    else:
                        print('invalid input')
        os.makedirs(save_dir, exist_ok=True)
    else:
        save_dir = None

    return log_dir, save_dir


def set_strategy(strategy):
    if strategy == 'ddp':
        strategy = pl.strategies.DDPStrategy()
    else:
        strategy = None
        
    return strategy


def set_callbacks(config, save_dir, monitor):
    callbacks = [
        CustomProgressBar(),
    ]
    if ((not config.no_eval) and
        monitor is not None and
        getattr(config, 'early_stopping_patience', -1) > 0):
        callbacks.append(CustomEarlyStopping(monitor=monitor, mode="min", patience=config.early_stopping_patience))

    if not config.no_save and save_dir is not None:
        # last checkpointing
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            dirpath=save_dir,
            filename=f'last',
            auto_insert_metric_name=False,
            every_n_epochs=1,
            save_top_k=1,
            save_last=False,
            monitor='epoch',
            mode='max',
        )
        checkpoint_callback.CHECKPOINT_JOIN_CHAR = "_"
        callbacks.append(checkpoint_callback)
        
        # best checkpointing
        if not (config.no_eval or monitor is None):
            checkpoint_callback = pl.callbacks.ModelCheckpoint(
                dirpath=save_dir,
                filename=f'best',
                auto_insert_metric_name=False,
                every_n_epochs=1,
                save_top_k=1,
                save_last=False,
                monitor=monitor,
                mode='min',
            )
            checkpoint_callback.CHECKPOINT_JOIN_CHAR = "_"
            callbacks.append(checkpoint_callback)
            
    return callbacks


def get_ckpt_path(config, verbose=True):
    if config.load_step == 0:
        ckpt_name = 'best.ckpt'
    elif config.load_step < 0:
        ckpt_name = 'last.ckpt'
    else:
        ckpt_name = f'step_{config.load_step:06d}.ckpt'
        
    load_path = os.path.join('experiments', config.load_dir, config.exp_name, 'checkpoints', ckpt_name)
    if not os.path.exists(load_path):
        raise FileNotFoundError(f'checkpoint ({load_path}) does not exists!')
            
    return load_path

        
class CustomProgressBar(TQDMProgressBar):
    def __init__(self, rescale_validation_batches=1):
        super().__init__()
        self.rescale_validation_batches = rescale_validation_batches

    def init_train_tqdm(self):
        """Override this to customize the tqdm bar for training."""
        bar = tqdm.tqdm(
            desc="Training",
            bar_format="{desc:<5}{percentage:3.0f}%|{bar:10}{r_bar}",
            initial=self.train_batch_idx,
            position=(2 * self.process_position),
            disable=self.is_disabled,
            leave=True,
            dynamic_ncols=True,
            file=sys.stdout,
            smoothing=0,
        )
        return bar

    def init_validation_tqdm(self):
        """Override this to customize the tqdm bar for validation."""
        # The main progress bar doesn't exist in `trainer.validate()`
        has_main_bar = self.trainer.state.fn != "validate"
        bar = tqdm.tqdm(
            desc="Validation",
            bar_format="{desc:<5}{percentage:3.0f}%|{bar:10}{r_bar}",
            position=(2 * self.process_position + has_main_bar),
            disable=self.is_disabled,
            leave=not has_main_bar,
            dynamic_ncols=True,
            file=sys.stdout,
        )
        return bar

    
class CustomTBLogger(TensorBoardLogger):
    @pl.utilities.rank_zero_only
    def log_metrics(self, metrics, step):
        metrics.pop('epoch', None)
        return super().log_metrics(metrics, step)

    
class CustomEarlyStopping(EarlyStopping):
    def _run_early_stopping_check(self, trainer):
        """Checks whether the early stopping condition is met and if so tells the trainer to stop the training."""
        logs = trainer.callback_metrics
        if self.monitor not in logs:
            should_stop = False
            reason = None
        else:
            current = logs[self.monitor].squeeze()
            should_stop, reason = self._evaluate_stopping_criteria(current)

        # stop every ddp process if any world process decides to stop
        should_stop = trainer.strategy.reduce_boolean_decision(should_stop)
        trainer.should_stop = trainer.should_stop or should_stop
        if should_stop:
            self.stopped_epoch = trainer.current_epoch
        if reason and self.verbose:
            self._log_info(trainer, reason, self.log_rank_zero_only)