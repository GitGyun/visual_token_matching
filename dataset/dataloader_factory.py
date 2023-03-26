import torch
from torch.utils.data import DataLoader
import os
from einops import rearrange, repeat

from .taskonomy import TaskonomyHybridDataset, TaskonomyContinuousDataset, TaskonomySegmentationDataset, TaskonomyFinetuneDataset
from .taskonomy_constants import TASKS, TASKS_GROUP_DICT, TASKS_GROUP_TRAIN, TASKS_GROUP_NAMES, \
                                 BUILDINGS, BUILDINGS_TRAIN, BUILDINGS_VALID, BUILDINGS_TEST, SEMSEG_CLASSES
from .utils import crop_arrays


base_sizes = {
    224: (256, 256)
}


def get_train_loader(config, pin_memory=True, verbose=True, get_support_data=False):
    '''
    Load training dataloader.
    '''
    # set dataset size
    if get_support_data:
        dset_size = config.shot
    elif config.no_eval:
        dset_size = config.n_steps*config.global_batch_size
    else:
        dset_size = config.val_iter*config.global_batch_size

    # compute common arguments
    common_kwargs = {
        'base_size': base_sizes[config.img_size],
        'img_size': (config.img_size, config.img_size),
        'dset_size': dset_size,
        'seed': config.seed + int(os.environ.get('LOCAL_RANK', 0)),
        'precision': config.precision,
        'root_dir': config.root_dir,
        'buildings': BUILDINGS_TRAIN,
    }

    # create dataset for episodic training
    if config.stage == 0:
        tasks = TASKS if config.task == 'all' else TASKS_GROUP_TRAIN[config.task_fold]
        if verbose:
            print(f'Loading tasks {", ".join(tasks)} in train split.')

        # create training dataset.
        train_data = TaskonomyHybridDataset(
            tasks=tasks,
            shot=config.shot,
            tasks_per_batch=config.max_channels,
            domains_per_batch=config.domains_per_batch,
            image_augmentation=config.image_augmentation,
            unary_augmentation=config.unary_augmentation,
            binary_augmentation=config.binary_augmentation,
            mixed_augmentation=config.mixed_augmentation,
            **common_kwargs,
        )
     # create dataset for fine-tuning or testing
    else:
        if config.task in ['', 'all']:
            raise ValueError("task should be specified for fine-tuning")

        train_data = TaskonomyFinetuneDataset(
            task=config.task,
            shot=config.shot,
            support_idx=config.support_idx,
            channel_idx=config.channel_idx,
            image_augmentation=(config.image_augmentation and config.task != 'normal'),
            **common_kwargs
        )
        if get_support_data:
            train_data.fix_seed = True
            if config.stage == 1:
                train_data.img_size = base_sizes[config.img_size]
            support_loader = DataLoader(train_data, batch_size=1, shuffle=False, drop_last=False)
            for support_data in support_loader:
                break

            return support_data


    # create training loader.
    train_loader = DataLoader(train_data, batch_size=(config.global_batch_size // torch.cuda.device_count()),
                              shuffle=False, pin_memory=pin_memory,
                              drop_last=True, num_workers=config.num_workers)
        
    return train_loader


def get_eval_loader(config, task, split='valid', channel_idx=-1, pin_memory=True, verbose=True):
    '''
    Load evaluation dataloader.
    '''
    # no crop for evaluation.
    img_size = base_size = base_sizes[config.img_size]
    
    # choose appropriate split.
    if split == 'train':
        buildings = BUILDINGS_TRAIN
    elif split == 'valid':
        buildings = BUILDINGS_VALID
    elif split == 'test':
        buildings = BUILDINGS_TEST
    elif split in BUILDINGS:
        buildings = [split]
        
    # evaluate some subset or the whole data.
    if config.n_eval_batches > 0:
        dset_size = config.n_eval_batches * config.eval_batch_size
    else:
        dset_size = -1
    
    # common arguments for both continuous and segmentation datasets.
    common_kwargs = {
        'root_dir': config.root_dir,
        'buildings': buildings,
        'dset_size': dset_size,
        'base_size': base_size,
        'img_size': img_size,
        'seed': int(os.environ.get('LOCAL_RANK', 0)),
        'precision': config.precision,
    }
    if verbose:
        if channel_idx < 0:
            print(f'Loading task {task} in {split} split.')
        else:
            print(f'Loading task {task}_{channel_idx} in {split} split.')
    
    # create appropriate dataset.
    if task == 'segment_semantic':
        assert channel_idx in SEMSEG_CLASSES
        eval_data = TaskonomySegmentationDataset(
            semseg_class=channel_idx,
            **common_kwargs
        )
    else:
        eval_data = TaskonomyContinuousDataset(
            task=task,
            channel_idx=channel_idx,
            **common_kwargs
        )

    # create dataloader.
    eval_loader = DataLoader(eval_data, batch_size=(config.eval_batch_size // torch.cuda.device_count()),
                             shuffle=False, pin_memory=pin_memory,
                             drop_last=False, num_workers=1)
    
    return eval_loader


def get_validation_loaders(config, verbose=True):
    '''
    Load validation loaders (of unseen images) for training tasks.
    '''
    if config.stage == 0:
        if config.task == 'all':
            train_tasks = TASKS_GROUP_DICT
        else:
            train_tasks = TASKS_GROUP_TRAIN[config.task_fold]
        loader_tag = 'mtrain_valid'
    else:
        if config.task in ['', 'all']:
            raise ValueError("task should be specified for fine-tuning")
        train_tasks = [config.task]
        loader_tag = 'mtest_valid'

    valid_loaders = {}
    for task in train_tasks:
        if task == 'segment_semantic':
            if config.channel_idx < 0:
                channels = SEMSEG_CLASSES
            else:
                channels = [config.channel_idx]
            for c in channels:
                valid_loaders[f'segment_semantic_{c}'] = get_eval_loader(config, task, 'valid', c, verbose=verbose)
        else:
            valid_loaders[task] = get_eval_loader(config, task, 'valid', verbose=verbose)
    
    return valid_loaders, loader_tag


def generate_support_data(config, data_path, split='train', support_idx=0, verbose=True):
    '''
    Generate support data for all tasks.
    '''
    if os.path.exists(data_path):
        support_data = torch.load(data_path)
    else:
        support_data = {}
    
    modified = False
    base_size = img_size = base_sizes[config.img_size]
    
    if split == 'train':
        buildings = BUILDINGS_TRAIN
    elif split == 'valid':
        buildings = BUILDINGS_VALID
    elif split == 'test':
        buildings = BUILDINGS_TEST
    else:
        raise ValueError(split)
    
    common_kwargs = {
        'root_dir': config.root_dir,
        'buildings': buildings,
        'base_size': base_size,
        'img_size': img_size,
        'seed': int(os.environ.get('LOCAL_RANK', 0)),
        'precision': 'fp32',
    }
    
    for task in TASKS_GROUP_NAMES:
        if task == 'segment_semantic':
            for c in SEMSEG_CLASSES:
                if f'segment_semantic_{c}' in support_data:
                    continue

                dset = TaskonomySegmentationDataset(
                    semseg_class=c,
                    **common_kwargs
                )
                dloader = DataLoader(dset, batch_size=config.shot, shuffle=False, num_workers=0)
                for idx, batch in enumerate(dloader):
                    if idx == support_idx:
                        break
        
                X, Y, M = batch
                
                T = Y.size(1)
                X = repeat(X, 'N C H W -> 1 T N C H W', T=T)
                Y = rearrange(Y, 'N T H W -> 1 T N 1 H W')
                M = rearrange(M, 'N T H W -> 1 T N 1 H W')
                X, Y, M = crop_arrays(X, Y, M, base_size=base_size, img_size=(config.img_size, config.img_size),
                                      random=False)
                
                t_idx = torch.tensor([[TASKS.index(f'segment_semantic_{c}')]])
                
                support_data[f'segment_semantic_{c}'] = (X, Y, M, t_idx)
                if verbose:
                    print(f'generated support data for task segment_semantic_{c}')
                modified = True
        else:
            if task in support_data:
                continue

            dset = TaskonomyContinuousDataset(
                task=task,
                **common_kwargs
            )
            
            dloader = DataLoader(dset, batch_size=config.shot, shuffle=False, num_workers=0)
            for idx, batch in enumerate(dloader):
                if idx == support_idx:
                    break

            X, Y, M = batch
            T = Y.size(1)
            X = repeat(X, 'N C H W -> 1 T N C H W', T=T)
            Y = rearrange(Y, 'N T H W -> 1 T N 1 H W')
            M = rearrange(M, 'N T H W -> 1 T N 1 H W')
            X, Y, M = crop_arrays(X, Y, M, base_size=base_size, img_size=(config.img_size, config.img_size),
                                    random=False)
            
            t_idx = torch.tensor([[TASKS.index(f'{task}_{c}') for c in range(len(TASKS_GROUP_DICT[task]))]])

            support_data[task] = (X, Y, M, t_idx)
            if verbose:
                print(f'generated support data for task {task}')
            modified = True

    if modified:
        torch.save(support_data, data_path)
            
    return support_data