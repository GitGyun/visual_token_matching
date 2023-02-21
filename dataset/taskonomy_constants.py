# Building Splits
BUILDINGS_TRAIN = ['allensville', 'beechwood', 'benevolence', 'coffeen', 'cosmos', 
                   'forkland', 'hanson', 'hiteman', 'klickitat', 'lakeville', 
                   'leonardo', 'lindenwood', 'marstons', 'merom', 'mifflinburg', 
                   'newfields', 'onaga', 'pinesdale', 'pomaria', 'ranchester', 
                   'shelbyville', 'stockman', 'tolstoy', 'wainscott', 'woodbine']
BUILDINGS_VALID = ['collierville', 'corozal', 'darden', 'markleeville', 'wiconisco']
BUILDINGS_TEST = ['ihlen', 'mcdade', 'muleshoe', 'noxapater', 'uvalda']
BUILDINGS = BUILDINGS_TRAIN + BUILDINGS_VALID + BUILDINGS_TEST

# Class Splits for Semantic Segmentation
SEMSEG_CLASSES = [2, 3, 4, 5, 6, 8, 9, 10, 12, 13, 15, 16]
SEMSEG_CLASS_RANGE = range(1, 17)

# Task Type Grouping
TASKS_SEMSEG = [f'segment_semantic_{c}' for c in SEMSEG_CLASSES]
TASKS_DEPTHE = [f'depth_euclidean_{c}' for c in range(5)]
TASKS_DEPTHZ = ['depth_zbuffer_0']
TASKS_EDGE2D = [f'edge_texture_{c}' for c in range(3)]
TASKS_EDGE3D = [f'edge_occlusion_{c}' for c in range(5)]
TASKS_KEYPOINTS2D = ['keypoints2d_0']
TASKS_KEYPOINTS3D = ['keypoints3d_0']
TASKS_NORMAL = [f'normal_{c}' for c in range(3)]
TASKS_RESHADING = ['reshading_0']
TASKS_CURVATURE = [f'principal_curvature_{c}' for c in range(2)]

# All Tasks
TASKS = TASKS_SEMSEG + TASKS_DEPTHE + TASKS_DEPTHZ + \
        TASKS_EDGE2D + TASKS_EDGE3D + TASKS_KEYPOINTS2D + TASKS_KEYPOINTS3D + \
        TASKS_NORMAL + TASKS_RESHADING + TASKS_CURVATURE

# Train and Test Tasks - can be splitted in other ways
N_SPLITS = 5
TASKS_GROUP_NAMES = ["segment_semantic", "normal", "depth_euclidean", "depth_zbuffer", "edge_texture", "edge_occlusion", "keypoints2d", "keypoints3d", "reshading", "principal_curvature"]
TASKS_GROUP_LIST = [TASKS_SEMSEG, TASKS_NORMAL, TASKS_DEPTHE, TASKS_DEPTHZ, TASKS_EDGE2D, TASKS_EDGE3D, TASKS_KEYPOINTS2D, TASKS_KEYPOINTS3D, TASKS_RESHADING, TASKS_CURVATURE]
TASKS_GROUP_DICT = {name: group for name, group in zip(TASKS_GROUP_NAMES, TASKS_GROUP_LIST)}

N_TASK_GROUPS = len(TASKS_GROUP_NAMES)
GROUP_UNIT = N_TASK_GROUPS // N_SPLITS

TASKS_GROUP_TRAIN = {}
TASKS_GROUP_TEST = {}
for split_idx in range(N_SPLITS):
    TASKS_GROUP_TRAIN[split_idx] = TASKS_GROUP_NAMES[:-GROUP_UNIT*(split_idx+1)] + (TASKS_GROUP_NAMES[-GROUP_UNIT*split_idx:] if split_idx > 0 else []) 
    TASKS_GROUP_TEST[split_idx] = TASKS_GROUP_NAMES[-GROUP_UNIT*(split_idx+1):-GROUP_UNIT*split_idx] if split_idx > 0 else TASKS_GROUP_NAMES[-GROUP_UNIT*(split_idx+1):] 

N_TASKS = len(TASKS)


CLASS_NAMES = ['bottle', 'chair', 'couch', 'plant',
               'bed', 'd.table', 'toilet', 'tv', 'microw', 
               'oven', 'toaster', 'sink', 'fridge', 'book',
               'clock', 'vase']