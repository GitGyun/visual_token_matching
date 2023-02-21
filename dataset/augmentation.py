import random
import math
import numpy as np
import torch
from torchvision.transforms.functional import gaussian_blur


def normalize(x):
    if x.max() == x.min():
        return x - x.min()
    else:
        return (x - x.min()) / (x.max() - x.min())


def linear_sample(p_range):
    if isinstance(p_range, float):
        return p_range
    else:
        return p_range[0] + random.random()*(p_range[1] - p_range[0])
    
    
def log_sample(p_range):
    if isinstance(p_range, float):
        return p_range
    else:
        return math.exp(math.log(p_range[0]) + random.random()*(math.log(p_range[1]) - math.log(p_range[0])))
    
    
def categorical_sample(p_range):
    if isinstance(p_range, (float, int)):
        return p_range
    else:
        return p_range[np.random.randint(len(p_range))]
    
    
def rand_bbox(size, lam):
    H, W = size
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


class Augmentation:
    pass


class RandomHorizontalFlip(Augmentation):
    def __init__(self):
        self.augmentation = lambda x: torch.flip(x, dims=[-1])
        
    def __str__(self):
        return 'RandomHorizontalFlip Augmentation'
        
    def __call__(self, *arrays, get_augs=False):
        if random.random() < 0.5:
            if len(arrays) == 1:
                if get_augs:
                    return self.augmentation(arrays[0]), self.augmentation
                else:
                    return self.augmentation(arrays[0])
            else:
                arrays_flipped = []
                for array in arrays:
                    arrays_flipped.append(self.augmentation(array))
                if get_augs:
                    return arrays_flipped, self.augmentation
                else:
                    return arrays_flipped
        else:
            if len(arrays) == 1:
                if get_augs:
                    return arrays[0], lambda x: x
                else:
                    return arrays[0]
            else:
                if get_augs:
                    return arrays, lambda x: x
                else:
                    return arrays
    
    
class RandomCompose(Augmentation):
    def __init__(self, augmentations, n_aug=2, p=0.5, verbose=False):
        assert len(augmentations) >= n_aug
        self.augmentations = augmentations
        self.n_aug = n_aug
        self.p = p
        self.verbose = verbose # for debugging
    
    def __call__(self, label, mask, get_augs=False):
        augmentations = [
            self.augmentations[i] 
            for i in np.random.choice(len(self.augmentations), size=self.n_aug, replace=False)
        ]
        
        for augmentation in augmentations:
            if random.random() < self.p:
                label, mask = augmentation(label, mask)
                if self.verbose:
                    print(augmentation)
            elif self.verbose:
                print('skipped')
            
        if get_augs:
            return label, mask, augmentations
        else:
            return label, mask
    

class RandomJitter(Augmentation):
    def __init__(self, brightness, contrast):
        self.brightness = brightness
        self.contrast = contrast
        
    def __str__(self):
        return f'RandomJitter Augmentation (brightness = {self.brightness}, contrast = {self.contrast})'
        
    def __call__(self, label, mask):
        brightness = linear_sample(self.brightness)
        contrast = linear_sample(self.contrast)
        
        alpha = 1 + contrast
        beta = brightness
        
        label = alpha * label + beta
        label = torch.clip(label, 0, 1)
        label = normalize(label)
        
        return label, mask
    
    
class RandomPolynomialTransform(Augmentation):
    def __init__(self, degree):
        self.degree = degree
        
    def __str__(self):
        return f'RandomPolynomialTransform Augmentation (degree = {self.degree})'
        
    def __call__(self, label, mask):
        degree = log_sample(self.degree)
        
        label = label.pow(degree)
        label = normalize(label)
        return label, mask


class RandomSigmoidTransform(Augmentation):
    def __init__(self, temperature):
        self.temperature = temperature
        
    def __str__(self):
        return f'RandomSigmoidTransform Augmentation (temperature = {self.temperature})'
    
    def __call__(self, label, mask):
        cast = False
        if label.dtype != torch.float32:
            dtype = label.dtype
            cast = True
            label = label.float()
        
        temperature = categorical_sample(self.temperature)
        
        label = torch.sigmoid(label / temperature)
        label = normalize(label)
        
        if cast:
            label = label.to(dtype)
        
        return label, mask


class RandomGaussianBlur(Augmentation):
    def __init__(self, kernel_size, sigma):
        self.kernel_size = kernel_size
        self.sigma = sigma
        
    def __str__(self):
        return f'RandomGaussianBlur Augmentation (kernel_size = {self.kernel_size}, sigma = {self.sigma})'
    
    def __call__(self, label, mask):
        cast = False
        if label.dtype != torch.float32:
            dtype = label.dtype
            cast = True
            label = label.float()
        
        kernel_size = [categorical_sample(self.kernel_size)]*2
        sigma = categorical_sample(self.sigma)
        
        label = gaussian_blur(label, kernel_size, sigma)
        label = normalize(label)
        
        if cast:
            label = label.to(dtype)
        
        return label, mask
    
    
class BinaryAugmentation(Augmentation):
    pass


class Mixup(BinaryAugmentation):
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        
    def __call__(self, label_1, label_2, mask_1, mask_2):
        lam = np.random.beta(self.alpha, self.alpha)
        label_mix = lam*label_1 + (1 - lam)*label_2
        mask_mix = torch.logical_and(mask_1, mask_2)
        
        return label_mix, mask_mix


class Cutmix(BinaryAugmentation):
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        
    def __call__(self, label_1, label_2, mask_1, mask_2):
        assert label_1.size() == label_2.size()
        
        lam = np.random.beta(self.alpha, self.alpha)
        bbx1, bby1, bbx2, bby2 = rand_bbox(label_1.size()[-2:], lam)
        
        label_mix = label_1.clone()
        label_mix[:, :, bbx1:bbx2, bby1:bby2] = label_2[:, :, bbx1:bbx2, bby1:bby2]
        mask_mix = mask_1.clone()
        mask_mix[:, :, bbx1:bbx2, bby1:bby2] = mask_2[:, :, bbx1:bbx2, bby1:bby2]

        return label_mix, mask_mix
     

FILTERING_AUGMENTATIONS = {
    'jitter': (RandomJitter, {"brightness": (-0.5, 0.5), 
                              "contrast": (-0.5, 0.5)}),
    'polynomial': (RandomPolynomialTransform, {"degree": (1.0/3, 3.0)}),
    'sigmoid': (RandomSigmoidTransform, {"temperature": [0.1, 0.2, 0.5, 2e5, 5e5, 1e6, 2e6]}),
    'gaussianblur': (RandomGaussianBlur, {"kernel_size": [9, 17, 33], 
                                          "sigma": [0.5, 1.0, 2.0, 5.0, 10.0]}),
}