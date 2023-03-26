import argparse
import yaml
from easydict import EasyDict

from dataset.taskonomy_constants import TASKS_GROUP_NAMES, TASKS_GROUP_TEST


def str2bool(v):
    if v == 'True' or v == 'true':
        return True
    elif v == 'False' or v == 'false':
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
        

# argument parser
parser = argparse.ArgumentParser()

# necessary arguments
parser.add_argument('--debug_mode', '-debug', default=False, action='store_true')
parser.add_argument('--continue_mode', '-cont', default=False, action='store_true')
parser.add_argument('--skip_mode', '-skip', default=False, action='store_true')
parser.add_argument('--no_eval', '-ne', default=False, action='store_true')
parser.add_argument('--no_save', '-ns', default=False, action='store_true')
parser.add_argument('--reset_mode', '-reset', default=False, action='store_true')

parser.add_argument('--stage', type=int, default=0, choices=[0, 1, 2])
parser.add_argument('--task', type=str, default='', choices=['', 'all'] + TASKS_GROUP_NAMES)
parser.add_argument('--task_fold', '-fold', type=int, default=None, choices=[0, 1, 2, 3, 4])
parser.add_argument('--exp_name', type=str, default='')
parser.add_argument('--exp_subname', type=str, default='')
parser.add_argument('--name_postfix', '-ptf', type=str, default='')
parser.add_argument('--save_postfix', '-sptf', type=str, default='')
parser.add_argument('--result_postfix', '-rptf', type=str, default='')
parser.add_argument('--load_step', '-ls', type=int, default=-1)

# optional arguments
parser.add_argument('--model', type=str, default=None, choices=['VTM'])
parser.add_argument('--seed', type=int, default=None)
parser.add_argument('--strategy', '-str', type=str, default=None)
parser.add_argument('--num_workers', '-nw', type=int, default=None)
parser.add_argument('--global_batch_size', '-gbs', type=int, default=None)
parser.add_argument('--eval_batch_size', '-ebs', type=int, default=None)
parser.add_argument('--n_eval_batches', '-neb', type=int, default=None)
parser.add_argument('--shot', type=int, default=None)
parser.add_argument('--max_channels', '-mc', type=int, default=None)
parser.add_argument('--support_idx', '-sid', type=int, default=None)
parser.add_argument('--channel_idx', '-cid', type=int, default=None)
parser.add_argument('--test_split', '-split', type=str, default=None)
parser.add_argument('--semseg_threshold', '-sth', type=float, default=None)

parser.add_argument('--image_augmentation', '-ia', type=str2bool, default=None)
parser.add_argument('--unary_augmentation', '-ua', type=str2bool, default=None)
parser.add_argument('--binary_augmentation', '-ba', type=str2bool, default=None)
parser.add_argument('--mixed_augmentation', '-ma', type=str2bool, default=None)
parser.add_argument('--image_backbone', '-ib', type=str, default=None)
parser.add_argument('--label_backbone', '-lb', type=str, default=None)
parser.add_argument('--n_attn_heads', '-nah', type=int, default=None)

parser.add_argument('--n_steps', '-nst', type=int, default=None)
parser.add_argument('--optimizer', '-opt', type=str, default=None, choices=['sgd', 'adam', 'adamw'])
parser.add_argument('--lr', type=float, default=None)
parser.add_argument('--lr_pretrained', '-lrp', type=float, default=None)
parser.add_argument('--lr_schedule', '-lrs', type=str, default=None, choices=['constant', 'sqroot', 'cos', 'poly'])
parser.add_argument('--early_stopping_patience', '-esp', type=int, default=None)

parser.add_argument('--log_dir', type=str, default=None)
parser.add_argument('--save_dir', type=str, default=None)
parser.add_argument('--load_dir', type=str, default=None)
parser.add_argument('--val_iter', '-viter', type=int, default=None)
parser.add_argument('--save_iter', '-siter', type=int, default=None)

args = parser.parse_args()


# load config file
if args.stage == 0:
    config_path = 'configs/train_config.yaml'
elif args.stage == 1:
    config_path = 'configs/finetune_config.yaml'
elif args.stage == 2:
    config_path = 'configs/test_config.yaml'

with open(config_path, 'r') as f:
    config = yaml.safe_load(f)
    config = EasyDict(config)

# copy parsed arguments
for key in args.__dir__():
    if key[:2] != '__' and getattr(args, key) is not None:
        setattr(config, key, getattr(args, key))

# retrieve data root
with open('data_paths.yaml', 'r') as f:
    path_dict = yaml.safe_load(f)
    config.root_dir = path_dict[config.dataset]
    
# for debugging
if config.debug_mode:
    config.n_steps = 10
    config.log_iter = 1
    config.val_iter = 5
    config.save_iter = 5
    if config.stage == 2:
        config.n_eval_batches = 2
    config.log_dir += '_debugging'
    if config.stage == 0:
        config.load_dir += '_debugging'
    if config.stage <= 1:
        config.save_dir += '_debugging'

# create experiment name
if config.exp_name == '':
    if config.stage == 0:
        if config.task == '':
            config.exp_name = f'{config.model}_fold:{config.task_fold}{config.name_postfix}'
        else:
            config.exp_name = f'{config.model}_task:{config.task}{config.name_postfix}'
    else:
        fold_dict = {}
        for fold in TASKS_GROUP_TEST:
            for task in TASKS_GROUP_TEST[fold]:
                fold_dict[task] = fold
        task_fold = fold_dict[config.task]
        config.exp_name = f'{config.model}_fold:{task_fold}{config.name_postfix}'