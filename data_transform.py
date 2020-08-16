import torch
import torchvision.transforms as transforms
import cv2
import numpy as np

'''
import kornia.augmentation as Ka
import kornia.filters as Kf
'''

import random
import torch.nn as nn


def round_up_to_odd(f):
    f = int(np.ceil(f))
    return f + 1 if f % 2 == 0 else f


class DataTransform(object):
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, sample):
        xi = self.transform(sample)
        xj = self.transform(sample)
        return xi, xj


class GaussianBlur(object):
    
    def __init__(self, kernel_size, min=0.1, max=2.0):
        self.min = min
        self.max = max
        self.kernel_size = kernel_size

    def __call__(self, sample):
        sample = np.array(sample)

        prob = random.random()
        if prob < 0.5:
            sigma = (self.max - self.min) * random.random() + self.min
            sample = cv2.GaussianBlur(sample, (self.kernel_size, self.kernel_size), sigma)

        return sample


class KfRandomGaussianBlur(nn.Module):

    def __init__(self, kernel_size, min=0.1, max=2.0):
        super().__init__()
        self.min = min
        self.max = max
        self.kernel_size = kernel_size

    def forward(self, sample):

        prob = random.random()
        if prob < 0.5:
            sigma = (self.max - self.min) * np.random.random_sample() + self.min
            return Kf.GaussianBlur2d((self.kernel_size, self.kernel_size), 
                (sigma, sigma))(sample)

        return sample


class KaRandomColorJitter(nn.Module):

    def __init__(self, strength, p = 0.5):
        super().__init__()
        self.strength = strength
        '''
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        '''
        self.p = p


    def forward(self, sample):

        prob = random.random()
        if prob < self.p:
            return Ka.ColorJitter(0.8 * self.strength, 0.8 * self.strength, 
                0.8 * self.strength, 0.2 * self.strength, same_on_batch = True)(sample)

        return sample


def get_spatial_transform(strength, crop_size, with_gauss_blur = False):
    
    color_jitter = transforms.ColorJitter(0.8 * strength, 0.8 * strength, 0.8 * strength, 0.2 * strength)

    if with_gauss_blur:
        data_transforms = transforms.Compose([transforms.RandomResizedCrop(size=crop_size),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.RandomApply([color_jitter], p=0.8),
                                            transforms.RandomGrayscale(p=0.2),
                                            GaussianBlur(kernel_size=round_up_to_odd(0.1 * crop_size)),
                                            transforms.ToTensor()])
    else:
        data_transforms = transforms.Compose([transforms.RandomResizedCrop(size=crop_size),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.RandomApply([color_jitter], p=0.8),
                                            transforms.RandomGrayscale(p=0.2),
                                            transforms.ToTensor()])

    return data_transforms



def get_improved_spatial_transform(strength, crop_size, with_gauss_blur = False):

    color_jitter = transforms.ColorJitter(0.8 * strength, 0.8 * strength, 0.8 * strength, 0.2 * strength)
    
    if with_gauss_blur:
        transforms_list = [transforms.RandomResizedCrop(size=crop_size),
                            transforms.RandomHorizontalFlip(),
                            transforms.RandomApply([color_jitter], p=0.8),
                            transforms.RandomGrayscale(p=0.2),
                            GaussianBlur(kernel_size=round_up_to_odd(0.1 * crop_size)),
                            transforms.ToTensor()]
    else:
        transforms_list = [transforms.RandomResizedCrop(size=crop_size),
                            transforms.RandomHorizontalFlip(),
                            transforms.RandomApply([color_jitter], p=0.8),
                            transforms.RandomGrayscale(p=0.2),
                            transforms.ToTensor()]

    def spatial_pipeline(img, seed = None):

        if seed is None:
            return transforms.Compose(transforms_list)

        random.seed(seed)
        img = transforms_list[0](img)
        img = transforms_list[1](img)

        new_seed = np.random.randint(np.iinfo('int32').max)
        random.seed(new_seed)
        img = transforms_list[2](img)
        img = transforms_list[3](img)
        if with_gauss_blur:
            img = transforms_list[4](img)
        tensor = transforms_list[-1](img)
        
        return tensor

    return spatial_pipeline



def kornia_get_data_transform(strength, crop_size):

    return nn.Sequential(Ka.RandomResizedCrop(size = (crop_size, crop_size), same_on_batch = True),
                         Ka.RandomHorizontalFlip(same_on_batch = True),
                         KaRandomColorJitter(strength, p = 0.8), 
                         Ka.RandomGrayscale(p=0.2, same_on_batch = True), 
                         KfRandomGaussianBlur(kernel_size = round_up_to_odd(0.1 * crop_size)))


def get_temporal_shift(frame_indices, step = 0):

    start_idx = step
    end_idx   = len(frame_indices) - step
    random_shift = random.choice(range(-step, step))
    indices_i = frame_indices[start_idx + random_shift: end_idx + random_shift]
    random_shift = random.choice(range(-step, step))
    indices_j = frame_indices[start_idx + random_shift: end_idx + random_shift]

    return indices_i, indices_j


def get_temporal_subsample(frame_indices, step = 0):

    even_idx = [i for i in range(len(frame_indices)) if i % 2 == 0]
    odd_idx  = [i for i in range(len(frame_indices)) if i not in even_idx]

    indices = np.array(frame_indices)

    if random.random() < 0.5:
        indices_i = indices[even_idx]
        indices_j = indices[odd_idx]
    else:
        indices_i = indices[odd_idx]
        indices_j = indices[even_idx]

    return indices_i.tolist(), indices_j.tolist()


def get_temporal_shuffle(frame_indices, step = 0):

    indices_i = random.sample(frame_indices, len(frame_indices))
    indices_j = random.sample(frame_indices, len(frame_indices))

    indices_i = indices_i[step: len(indices_i) - step]
    indices_j = indices_j[step: len(indices_j) - step]

    return indices_i, indices_j


def get_temporal_reverse(frame_indices, step = 0):

    indices_i = frame_indices
    indices_j = list(reversed(frame_indices))

    indices_i = indices_i[step: len(indices_i) - step]
    indices_j = indices_j[step: len(indices_j) - step]    

    return indices_i, indices_j


def get_temporal_transform(transform_type):

    if transform_type == "None":
        return None

    temporal_transform_dict = {
    'shift': get_temporal_shift,
    'drop': get_temporal_subsample,
    'shuffle': get_temporal_shuffle,
    'reverse': get_temporal_reverse
    }


    return temporal_transform_dict[transform_type]