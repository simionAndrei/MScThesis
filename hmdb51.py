import torch
import torch.utils.data as data
from PIL import Image
import os
import math
import functools
import json
import copy

import numpy as np
import random


def load_value_file(file_path):
    with open(file_path, 'r') as input_file:
        value = float(input_file.read().rstrip('\n\r'))

    return value


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def accimage_loader(path):
    try:
        import accimage
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def get_default_image_loader():
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader
    else:
        return pil_loader


def video_loader(video_dir_path, frame_indices, image_loader):
    video = []
    for i in frame_indices:
        image_path = os.path.join(video_dir_path, 'image_{:05d}.jpg'.format(i))
        if os.path.exists(image_path):
            video.append(image_loader(image_path))
        else:
            return video

    return video


def get_default_video_loader():
    image_loader = get_default_image_loader()
    return functools.partial(video_loader, image_loader=image_loader)


def load_annotation_data(data_file_path):
    with open(data_file_path, 'r') as data_file:
        return json.load(data_file)


def get_class_labels(data):
    class_labels_map = {}
    index = 0
    for class_label in data['labels']:
        class_labels_map[class_label] = index
        index += 1
    return class_labels_map


def get_video_names_and_annotations(data, subset):
    video_names = []
    annotations = []

    for key, value in data['database'].items():
        this_subset = value['subset']
        if this_subset == subset:
            label = value['annotations']['label']
            video_names.append('{}/{}'.format(label, key))
            annotations.append(value['annotations'])

    return video_names, annotations


def make_dataset(root_path, annotation_path, subset, n_samples_for_each_video,
                 sample_duration):
    data = load_annotation_data(annotation_path)
    video_names, annotations = get_video_names_and_annotations(data, subset)
    class_to_idx = get_class_labels(data)
    idx_to_class = {}
    for name, label in class_to_idx.items():
        idx_to_class[label] = name

    dataset = []
    for i in range(len(video_names)):
        if i % 1000 == 0:
            print('dataset loading [{}/{}]'.format(i, len(video_names)))

        video_path = os.path.join(root_path, video_names[i])
        if not os.path.exists(video_path):
            continue

        n_frames_file_path = os.path.join(video_path, 'n_frames')
        n_frames = int(load_value_file(n_frames_file_path))
        if n_frames <= 0:
            continue

        begin_t = 1
        end_t = n_frames
        sample = {
            'video': video_path,
            'segment': [begin_t, end_t],
            'n_frames': n_frames,
            'video_id': video_names[i].split('/')[1]
        }
        if len(annotations) != 0:
            sample['label'] = class_to_idx[annotations[i]['label']]
        else:
            sample['label'] = -1

        if n_samples_for_each_video == 1:
            sample['frame_indices'] = list(range(1, n_frames + 1))
            dataset.append(sample)
        else:
            if n_samples_for_each_video > 1:
                step = max(1,
                           math.ceil((n_frames - 1 - sample_duration) /
                                     (n_samples_for_each_video - 1)))
            else:
                step = sample_duration
            for j in range(1, n_frames, step):
                sample_j = copy.deepcopy(sample)
                sample_j['frame_indices'] = list(
                    range(j, min(n_frames + 1, j + sample_duration)))
                dataset.append(sample_j)

    return dataset, idx_to_class


class HMDB51(data.Dataset):
    """
    Args:
        root (string): Root directory path.
        spatial_transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        temporal_transform (callable, optional): A function/transform that  takes in a list of frame indices
            and returns a transformed version
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an video given its path and frame indices.
     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(self,
                 root_path,
                 annotation_path,
                 subset,
                 n_samples_for_each_video=1,
                 sampling_method=None,
                 sample_duration=16,
                 get_loader=get_default_video_loader,
                 stack_clip=False,
                 is_simclr_transform=False,
                 apply_same_per_clip=False,
                 spatial_transform=None,
                 temporal_transform=None,
                 temporal_step=0):

        self.data, self.class_names = make_dataset(
            root_path, annotation_path, subset, n_samples_for_each_video,
            sample_duration)

        self.sampling_method = sampling_method
        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        
        self.stack_clip = stack_clip
        self.is_simclr_transform = is_simclr_transform
        self.apply_same_per_clip = apply_same_per_clip
        self.temporal_step = temporal_step
        self.loader = get_loader()

    def __getitem__(self, index):
        
        path = self.data[index]['video']

        frame_indices = self.data[index]['frame_indices']
        if self.sampling_method is not None:
            frame_indices = self.sampling_method(frame_indices)
    
        if self.is_simclr_transform:

            seed1 = np.random.randint(np.iinfo('int32').max)
            seed2 = np.random.randint(np.iinfo('int32').max)

            if self.temporal_transform is not None:

                indices_i, indices_j = self.temporal_transform(frame_indices, self.temporal_step)

                clip_i = self.loader(path, indices_i)
                clip_j = self.loader(path, indices_j)
            
                clip = []
                for img_i, img_j in zip(clip_i, clip_j):
                    if self.apply_same_per_clip:
                        random.seed(seed1)
                    img_i = self.spatial_transform(img_i)

                    if self.apply_same_per_clip:
                        random.seed(seed2)
                    img_j = self.spatial_transform(img_j)

                    clip.append(torch.stack([img_i, img_j]))
            else:
                clip = self.loader(path, frame_indices)
                new_clip = []
                for img in clip:
                    if self.apply_same_per_clip:
                        random.seed(seed1)
                    img_i = self.spatial_transform(img)

                    if self.apply_same_per_clip:
                        random.seed(seed2)
                    img_j = self.spatial_transform(img)
                    
                    new_clip.append(torch.stack([img_i, img_j]))
                clip = new_clip
        else:
            clip = self.loader(path, frame_indices)

            '''
            seed = np.random.randint(np.iinfo('int32').max)
            new_clip = []
            for img in clip:
                if self.apply_same_per_clip:
                    random.seed(seed)
                new_clip.append(self.spatial_transform(img))
            clip = new_clip
            '''
            clip = [self.spatial_transform(img) for img in clip]

        if self.stack_clip and not self.is_simclr_transform:
            clip = torch.stack(clip, 0).permute(1, 0, 2, 3)

        if self.stack_clip and self.is_simclr_transform:
            clip = torch.stack(clip, 0).permute(1, 2, 0, 3, 4)

        target = self.data[index]
        label = target['label']
        
        return clip, label

    def __len__(self):
        return len(self.data)


def get_hmdb_dataset(video_path, annotation_path, dataset_type,
    sampling_method, spatial_transform, temporal_transform, temporal_step = 0,
    stack_clip = False, is_simclr_transform = False, apply_same_per_clip = False):

    data = HMDB51(video_path, annotation_path, dataset_type,
        sampling_method = sampling_method, spatial_transform = spatial_transform,
        temporal_transform = temporal_transform, temporal_step = temporal_step, 
        stack_clip = stack_clip, is_simclr_transform = is_simclr_transform,
        apply_same_per_clip = apply_same_per_clip)

    return data