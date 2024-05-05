from torch.utils import data
import numpy as np
import torch
import random
from scipy import ndimage


class DatasetBCP_T_2D(data.Dataset):
    def __init__(self, voxels, labels, mode='cc', is_test=False):
        super().__init__()
        self.voxels = [self.preprocessing(voxel) for voxel in voxels]
        self.labels = labels
        self.image_shape = voxels[0].shape
        self.mode = mode
        self.is_test = is_test

    def __len__(self):
        return len(self.voxels)

    def _normalize(self, voxel):
        voxel = (voxel - np.min(voxel)) / \
            (np.max(voxel) - np.min(voxel))  # 0-1
        voxel = (voxel * 2) - 1  # -1-1
        return voxel

    def preprocessing(self, voxel):
        nonzero = voxel[voxel > 0]
        voxel = np.clip(voxel, 0, np.mean(nonzero)+np.std(nonzero)*4)
        voxel = self._normalize(voxel)
        return voxel

    def random_stride(self, image, slide):
        image = np.pad(image, [(slide, slide), (slide, slide)],
                       'constant', constant_values=image.min())
        h, w = image.shape
        h1, w1 = self.image_shape[0], self.image_shape[1]
        x_l = np.random.randint(0, w - w1)
        x_r = x_l + w1
        y_l = np.random.randint(0, h - h1)
        y_r = y_l + h1
        image = image[y_l:y_r, x_l:x_r]
        return image

    def random_rotate_2d(self, image, angle_range):
        angle_deg = np.random.randint(-angle_range, angle_range)
        image = ndimage.rotate(
            image, angle_deg, mode="nearest", reshape=False, order=1, axes=(0, 1))
        return image

    def __getitem__(self, index):
        voxel = self.voxels[index]
        label = self.labels[index]

        if self.mode == 'cc':  # Corpus callosum slice range
            a = 170
            b = 220
        elif self.mode == 'brain':  # Brain tissue
            a = 100
            b = 270
        else:  # All slices
            a = 0
            b = 319

        if self.is_test:  # return all slices in the range when testing
            image = voxel[:, :, a:b].transpose(2, 0, 1)
        else:  # randomly choose one slice for training
            index = random.randint(a, b)
            image = voxel[:, :, index]

        if not self.is_test:  # data augmentation
            if random.random() < 0.8:
                image = self.random_stride(image, 20)
            if random.random() < 0.8:
                image = self.random_rotate_2d(image, 20)
            image = image[np.newaxis, :, :]

        return torch.from_numpy(image), torch.tensor(label, dtype=torch.float32)
