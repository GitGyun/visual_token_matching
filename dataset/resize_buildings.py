from multiprocessing import Pool
import os
import torch
import tqdm
import yaml
from PIL import Image

building_list = [
    'allensville',
    'beechwood',
    'benevolence',
    'collierville',
    'coffeen',
    'corozal',
    'cosmos',
    'darden',
    'forkland',
    'hanson',
    'hiteman',
    'ihlen',
    'klickitat',
    'lakeville',
    'leonardo',
    'lindenwood',
    'markleeville',
    'marstons',
    'mcdade',
    'merom',
    'mifflinburg',
    'muleshoe',
    'newfields',
    'noxapater',
    'onaga',
    'pinesdale',
    'pomaria',
    'ranchester',
    'shelbyville',
    'stockman',
    'tolstoy',
    'uvalda',
    'wainscott',
    'wiconisco',
    'woodbine',
]

task_list = [
    'rgb',
    'normal',
    'depth_euclidean',
    'depth_zbuffer',
    'edge_occlusion',
    'keypoints2d',
    'keypoints3d',
    'reshading',
    'principal_curvature',
    'segment_semantic'
]


def resize(args):
    load_path, save_path, mode = args
    try:
        img = Image.open(load_path)
        img = img.resize(size, mode)
        img.save(save_path)
        return None
    except Exception as e:
        print(e)
        return load_path


if __name__ == "__main__":
    verbose = True
    size = (256, 256)
    split = "tiny"
    n_threads = 20

    with open('data_paths.yaml', 'r') as f:
        path_dict = yaml.safe_load(f)
        load_root = save_root = path_dict['taskonomy']

    load_dir = os.path.join(load_root, split)
    assert os.path.isdir(load_dir)
    '''
    load_dir
    |--building
       |--task
          |--file
    '''
    save_dir = os.path.join(save_root, f"{split}_{size[0]}_merged")
    os.makedirs(save_dir, exist_ok=True)
    '''
    save_dir
    |--task
       |--file
    '''

    args = []
    print("creating args...")
    for b_idx, building in enumerate(building_list):
        assert os.path.isdir(os.path.join(load_dir, building))
        for task in task_list:
            mode = Image.NEAREST if task == "segment_semantic" else Image.BILINEAR
            if b_idx == 0:
                os.makedirs(os.path.join(save_dir, task), exist_ok=True)

            load_names = os.listdir(os.path.join(load_dir, building, task))
            load_paths = [os.path.join(load_dir, building, task, load_name) for load_name in load_names]
            save_paths = [os.path.join(save_dir, task, f'{building}_{load_name}') for load_name in load_names]
            modes = [mode]*len(load_names)
            args += list(zip(load_paths, save_paths, modes))

    fail_list = []
    pool = Pool(n_threads)
    total = len(args)
    pbar = tqdm.tqdm(total=total, bar_format="{desc:<5}{percentage:3.0f}%|{bar:10}{r_bar}")
    for fail_path in pool.imap(resize, args):
        if fail_path is not None:
            fail_list += [fail_path]
        pbar.update()
    pbar.close()

    torch.save(fail_list, "fail_list.pth")

    pool.close()
    pool.join()
