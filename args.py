import argparse
import yaml


def str2bool(v):
    if v == 'True' or v == 'true':
        return True
    elif v == 'False' or v == 'false':
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
        

# argument parser
parser = argparse.ArgumentParser()

# environment arguments
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--precision', '-prc', type=str, default='bf16', choices=['fp32', 'fp16', 'bf16']) 
parser.add_argument('--strategy', '-str', type=str, default='ddp', choices=['none', 'ddp']) 
parser.add_argument('--debug_mode', '-debug', default=False, action='store_true')
parser.add_argument('--continue_mode', '-cont', default=False, action='store_true')
parser.add_argument('--skip_mode', '-skip', default=False, action='store_true')
parser.add_argument('--no_eval', '-ne', default=False, action='store_true')
parser.add_argument('--no_save', '-ns', default=False, action='store_true')
parser.add_argument('--reset_mode', '-reset', default=False, action='store_true')
parser.add_argument('--profile_mode', '-prof', default=False, action='store_true')
parser.add_argument('--sanity_check', '-sc', default=False, action='store_true')

# data arguments
parser.add_argument('--dataset', type=str, default='taskonomy', choices=['taskonomy'])
parser.add_argument('--task', type=str, default='', choices=['', 'all'])
parser.add_argument('--task_fold', '-fold', type=int, default=0, choices=[0, 1, 2, 3, 4]) 

parser.add_argument('--num_workers', '-nw', type=int, default=8)
parser.add_argument('--global_batch_size', '-gbs', type=int, default=8)
parser.add_argument('--max_channels', '-mc', type=int, default=5)
parser.add_argument('--shot', type=int, default=4)
parser.add_argument('--domains_per_batch', '-dpb', type=int, default=2)
parser.add_argument('--eval_batch_size', '-ebs', type=int, default=8)
parser.add_argument('--n_eval_batches', '-neb', type=int, default=10)

parser.add_argument('--img_size', type=int, default=224, choices=[224])
parser.add_argument('--image_augmentation', '-ia', type=str2bool, default=True)
parser.add_argument('--unary_augmentation', '-ua', type=str2bool, default=True)
parser.add_argument('--binary_augmentation', '-ba', type=str2bool, default=True)
parser.add_argument('--mixed_augmentation', '-ma', type=str2bool, default=True)

# model arguments
parser.add_argument('--model', type=str, default='VTM', choices=['VTM'])
parser.add_argument('--image_backbone', '-ib', type=str, default='beit_base_patch16_224_in22k')
parser.add_argument('--label_backbone', '-lb', type=str, default='vit_base_patch16_224')
parser.add_argument('--image_encoder_weights', '-iew', type=str, default='imagenet', choices=['none', 'imagenet'])
parser.add_argument('--label_encoder_weights', '-lew', type=str, default='none', choices=['none', 'imagenet'])
parser.add_argument('--n_attn_heads', '-nah', type=int, default=4)
parser.add_argument('--n_attn_layers', '-nal', type=int, default=1)
parser.add_argument('--attn_residual', '-ar', type=str2bool, default=True)
parser.add_argument('--out_activation', '-oa', type=str, default='sigmoid', choices=['sigmoid', 'clip', 'none'])
parser.add_argument('--drop_rate', '-dr', type=float, default=0.0)
parser.add_argument('--drop_path_rate', '-dpr', type=float, default=0.1)
parser.add_argument('--bitfit', '-bf', type=str2bool, default=True)
parser.add_argument('--semseg_threshold', '-th', type=float, default=0.2)

# training arguments
parser.add_argument('--n_steps', '-nst', type=int, default=300000)
parser.add_argument('--optimizer', '-opt', type=str, default='adam', choices=['sgd', 'adam', 'adamw', 'fadam', 'dsadam'])
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--lr_pretrained', '-lrp', type=float, default=1e-5)
parser.add_argument('--lr_schedule', '-lrs', type=str, default='poly', choices=['constant', 'sqroot', 'cos', 'poly'])
parser.add_argument('--lr_warmup', '-lrw', type=int, default=5000)
parser.add_argument('--lr_warmup_scale', '-lrws', type=float, default=0.)
parser.add_argument('--weight_decay', '-wd', type=float, default=0.)
parser.add_argument('--lr_decay_degree', '-ldd', type=float, default=0.9)
parser.add_argument('--temperature', '-temp', type=float, default=-1.)
parser.add_argument('--reg_coef', '-rgc', type=float, default=1.)
parser.add_argument('--mask_value', '-mv', type=float, default=-1.)

# logging arguments
parser.add_argument('--log_dir', type=str, default='TRAIN')
parser.add_argument('--save_dir', type=str, default='')
parser.add_argument('--load_dir', type=str, default='')
parser.add_argument('--exp_name', type=str, default='')
parser.add_argument('--name_postfix', '-ptf', type=str, default='')
parser.add_argument('--log_iter', '-li', type=int, default=100)
parser.add_argument('--val_iter', '-vi', type=int, default=10000)
parser.add_argument('--save_iter', '-si', type=int, default=10000)
parser.add_argument('--load_step', '-ls', type=int, default=-1)

config = parser.parse_args()


# retrieve data root
with open('data_paths.yaml', 'r') as f:
    path_dict = yaml.safe_load(f)
    config.root_dir = path_dict[config.dataset]
if config.save_dir == '':
    config.save_dir = config.log_dir
if config.load_dir == '':
    config.load_dir = config.log_dir
    
# for debugging
if config.debug_mode:
    config.n_steps = 10
    config.log_iter = 1
    config.val_iter = 5
    config.save_iter = 5
    config.n_eval_batches = 4
    config.log_dir += '_debugging'
    config.save_dir += '_debugging'
    config.load_dir += '_debugging'
    

# model-specific hyper-parameters
config.n_levels = 4
    
# adjust backbone names
if config.image_backbone in ['beit_base', 'beit_large']:
    config.image_backbone += '_patch16_224_in22k'
if config.image_backbone in ['vit_tiny', 'vit_small', 'vit_base', 'vit_large']:
    config.image_backbone += '_patch16_224'
if config.label_backbone in ['vit_tiny', 'vit_small', 'vit_base', 'vit_large']:
    config.label_backbone += '_patch16_224'
