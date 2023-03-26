import pytorch_lightning as pl
from pytorch_lightning.callbacks.progress.tqdm_progress import TQDMProgressBar
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.plugins.io.torch_plugin import TorchCheckpointIO

import os
import sys
import shutil
import random
import tqdm

import numpy as np
import torch

from .trainer import LightningTrainWrapper
from dataset.taskonomy_constants import TASKS, TASKS_GROUP_DICT


def configure_experiment(config, model):
    # set seeds
    set_seeds(config.seed)
    
    # set directories
    log_dir, save_dir = set_directories(config,
                                        exp_name=config.exp_name,
                                        exp_subname=(config.exp_subname if config.stage >= 1 else ''),
                                        create_save_dir=(config.stage <= 1))

    # create lightning callbacks, logger, and checkpoint plugin
    if config.stage <= 1:
        callbacks = set_callbacks(config, save_dir, config.monitor, ptf=config.save_postfix)
        logger = CustomTBLogger(log_dir, name='', version='', default_hp_metric=False)
    else:
        callbacks = set_callbacks(config, save_dir)
        logger = None
    
    # parse precision
    precision = int(config.precision.strip('fp')) if config.precision in ['fp16', 'fp32'] else config.precision
        
    # choose accelerator
    strategy = set_strategy(config.strategy)

    # choose plugins
    if config.stage == 1:
        plugins = [CustomCheckpointIO([f'model.{name}' for name in model.model.bias_parameter_names()])]
    else:
        plugins = None
    
    return logger, log_dir, callbacks, precision, strategy, plugins


def set_seeds(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def set_directories(config, root_dir='experiments', exp_name='', log_dir='logs', save_dir='checkpoints',
                    create_log_dir=True, create_save_dir=True, dir_postfix='', exp_subname=''):
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
        if exp_subname != '':
            log_root = os.path.join(log_root, exp_subname)
            os.makedirs(log_root, exist_ok=True)
        log_dir = os.path.join(log_root, log_dir)

        # reset the logging directory if exists
        if config.stage == 0 and os.path.exists(log_dir) and not (config.continue_mode or config.skip_mode):
            shutil.rmtree(log_dir)
        os.makedirs(log_dir, exist_ok=True)
    else:
        log_dir = None

    # set saving directory
    if create_save_dir:
        save_root = os.path.join(root_dir, config.save_dir, exp_name + dir_postfix)
        if exp_subname != '':
            save_root = os.path.join(save_root, exp_subname)
        save_dir = os.path.join(save_root, save_dir)

        # create the saving directory if checkpoint doesn't exist or in skipping mode,
        # otherwise ask user to reset it
        if config.stage == 0 and os.path.exists(save_dir) and int(os.environ.get('LOCAL_RANK', 0)) == 0:
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


def set_callbacks(config, save_dir, monitor=None, ptf=''):
    callbacks = [
        CustomProgressBar(),
    ]
    if ((not config.no_eval) and
        monitor is not None and
        config.early_stopping_patience > 0):
        callbacks.append(CustomEarlyStopping(monitor=monitor, mode="min", patience=config.early_stopping_patience))

    if not config.no_save and save_dir is not None:
        # last checkpointing
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            dirpath=save_dir,
            filename=f'last{ptf}',
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
                filename=f'best{ptf}',
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


def get_ckpt_path(load_dir, exp_name, load_step, exp_subname='', save_postfix=''):
    if load_step == 0:
        ckpt_name = f'best{save_postfix}.ckpt'
    elif load_step < 0:
        ckpt_name = f'last{save_postfix}.ckpt'
    else:
        ckpt_name = f'step_{load_step:06d}.ckpt'
        
    load_path = os.path.join('experiments', load_dir, exp_name, exp_subname, 'checkpoints', ckpt_name)
    if not os.path.exists(load_path):
        raise FileNotFoundError(f"checkpoint ({load_path}) does not exists!")
            
    return load_path


def copy_values(config_new, config_old):
    for key in config_new.__dir__():
        if key[:2] != '__':
            setattr(config_old, key, getattr(config_new, key))


def load_ckpt(ckpt_path, config_new=None):
    ckpt = torch.load(ckpt_path)
    state_dict = ckpt['state_dict']
    config = ckpt['hyper_parameters']['config']

    # merge config
    if config_new is not None:
        copy_values(config_new, config)

    return state_dict, config


def select_task_specific_parameters(config, model, state_dict):
    if config.channel_idx < 0:
        t_idx = torch.tensor([TASKS.index(task) for task in TASKS_GROUP_DICT[config.task]])
    else:
        t_idx = torch.tensor([TASKS.index(f'{config.task}_{config.channel_idx}')])

    # for fine-tuning
    bias_parameters = [f'model.{name}' for name in model.model.bias_parameter_names()]
    for key in state_dict.keys():
        if key in bias_parameters:
            state_dict[key] = state_dict[key][t_idx]


def load_model(config, verbose=True):
    load_path = None

    # create trainer for episodic training
    if config.stage == 0:
        model = LightningTrainWrapper(config, verbose=verbose)
        if config.continue_mode:
            load_path = get_ckpt_path(config.load_dir, config.exp_name, -1, save_postfix=config.save_postfix)

    # create trainer for fine-tuning or evaluation
    else:
        # load meta-trained checkpoint
        ckpt_path = get_ckpt_path(config.load_dir, config.exp_name, 0)
        state_dict, config = load_ckpt(ckpt_path, config)
        
        model = LightningTrainWrapper(config=config, verbose=verbose)
        # select task-specific parameters for test task
        if config.stage == 1:
            select_task_specific_parameters(config, model, state_dict)
        # load fine-tuned checkpoint
        else:
            ft_ckpt_path = get_ckpt_path(config.save_dir, config.exp_name, 0, config.exp_subname, config.save_postfix)
            ft_state_dict, _ = load_ckpt(ft_ckpt_path)
            for key in ft_state_dict:
                state_dict[key] = ft_state_dict[key]
                    
        print(model.load_state_dict(state_dict))
        if verbose:
            print(f'meta-trained checkpoint loaded from {ckpt_path}')
            if config.stage == 2:
                print(f'fine-tuned checkpoint loaded from {ft_ckpt_path}')

    return model, load_path

        
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
    
    def init_test_tqdm(self):
        """Override this to customize the tqdm bar for testing."""
        bar = tqdm.tqdm(
            desc="Testing",
            bar_format="{desc:<5}{percentage:3.0f}%|{bar:10}{r_bar}",
            position=(2 * self.process_position),
            disable=self.is_disabled,
            leave=True,
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


class CustomCheckpointIO(TorchCheckpointIO):
    def __init__(self, save_parameter_names):
        self.save_parameter_names = save_parameter_names
    
    def save_checkpoint(self, checkpoint, path, storage_options=None):
        # store only task-specific parameters
        state_dict = checkpoint['state_dict']
        state_dict = {key: value for key, value in state_dict.items() if key in self.save_parameter_names}
        checkpoint['state_dict'] = state_dict
        
        super().save_checkpoint(checkpoint, path, storage_options)
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
