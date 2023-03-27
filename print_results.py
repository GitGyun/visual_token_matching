import torch
import pandas as pd
import math
import os
from dataset.taskonomy_constants import TASKS_GROUP_TEST, SEMSEG_CLASSES
from train.miou_fss import AverageMeter
import argparse


def create_table(model, tasks, ptf, print_failure=False):
    result_root = os.path.join('experiments', args.result_dir)
        
    df = pd.DataFrame(index=[model], columns=[task_tags[task] for task in tasks])
    for task in tasks:
        task_tag = task_tags[task]
        exp_name = f'{model}_fold:{fold_dict[task]}{ptf}'
        exp_dir = os.path.join(result_root, exp_name)
        if not os.path.exists(exp_dir):
            continue
        if task == 'segment_semantic':
            success = True
            average_meter = AverageMeter(range(len(SEMSEG_CLASSES)))
            for i, c in enumerate(SEMSEG_CLASSES):
                result_name = f'result_task:{task}_{c}_split:{args.test_split}.pth'
                result_path = os.path.join(result_root, exp_name, 'logs', result_name)
                try:
                    average_meter_c = torch.load(result_path, map_location='cpu')
                    assert isinstance(average_meter_c, AverageMeter)
                except:
                    success = False
                    break

                average_meter.intersection_buf[:, i] += average_meter_c.intersection_buf[:, 0].cpu()
                average_meter.union_buf[:, i] += average_meter_c.union_buf[:, 0].cpu()

            if success:
                df.loc[model][task_tag] = average_meter.compute_iou()[0].cpu().item()
            elif print_failure:
                print(result_path)
        else:
            result_name = f'result_task:{task}_split:{args.test_split}.pth'
            result_path = os.path.join(result_root, exp_name, 'logs', result_name)
            if os.path.exists(result_path):
                result = torch.load(result_path)
                df.loc[model][task_tag] = result
            elif print_failure:
                print(result_path)

    return df


if __name__ == '__main__':
    from dataset.taskonomy_constants import TASKS_GROUP_NAMES

    parser = argparse.ArgumentParser()
    parser.add_argument('--result_dir', type=str, default='TEST')
    parser.add_argument('--test_split', type=str, default='muleshoe')
    parser.add_argument('--model', type=str, default='VTM')
    parser.add_argument('--name_postfix', '-ptf', type=str, default='')
    parser.add_argument('--task', type=str, default='all', choices=['all'] + TASKS_GROUP_NAMES)
    args = parser.parse_args()

    task_tags = {
        'segment_semantic': 'Semantic Segmentation (mIoU ↑)',
        'normal': 'Surface Normal (mErr ↓)',
        'depth_euclidean': 'Euclidean Distance (RMSE ↓)',
        'depth_zbuffer': 'Zbuffer Depth (RMSE ↓)',
        'edge_texture': 'Texture Edge (RMSE ↓)',
        'edge_occlusion': 'Occlusion Edge (RMSE ↓)',
        'keypoints2d': '2D Keypoints (RMSE ↓)',
        'keypoints3d': '3D Keypoints (RMSE ↓)',
        'reshading': 'Reshading (RMSE ↓)',
        'principal_curvature': 'Principal Curvature (RMSE ↓)',
    }
    fold_dict = {}
    for fold in TASKS_GROUP_TEST:
        for task in TASKS_GROUP_TEST[fold]:
            fold_dict[task] = fold

    if args.task == 'all':
        tasks = TASKS_GROUP_NAMES
    else:
        tasks = [args.task]

    pd.set_option('max_columns', None)
    df = create_table(args.model, tasks, args.name_postfix, print_failure=False)
    print(df)
