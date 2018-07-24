import pandas as pd
import numpy as np
import os
from skimage import io, transform

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils


class CustomDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.data_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir,
                                self.data_frame.iloc[idx, 0])
        image = io.imread(img_name)
        landmarks = self.data_frame.iloc[idx, 1:].as_matrix()
        landmarks = landmarks.astype('float').reshape(-1, 2)
        sample = {'image': image, 'landmarks': landmarks}

        if self.transform:
            sample = self.transform(sample)

        return sample

class DataHandler:
    def __init__(self, batch_size, image_size=None):
        self.batch_size = batch_size
        self.image_size = image_size


    def get_csv(self, path, p_val=0.2, p_test = 0.2):
    '''
        Takes path to csv file and breaks file down into appropriate training,
        validation, and test sections that can be accessed via instance variabless
    '''


    def get_data(self, path):
    '''
        Takes path to data folders and places training, validation, and test data
        in accessible instance variables
    '''
    if self.image_size is not None:
        train_set=torchvision.datasets.ImageFolder(root='/path/to/your/data/trn', transform=generic_transform)
        test_set=torchvision.datasets.ImageFolder(root='/path/to/your/data/val', transform=generic_transform)
    else:


    def batch_load(self):
    '''
        Generator function to load minibatches to network
    '''


    def sample(self):
    '''
        If images, loads single examples
    '''



    def transform(self, tforms):
    '''
        Performs entered transformations on data if images
    '''


# split off validation set
def get_val_idx(ds_size, cv_percent=0.2):
    return np.random.choice(ds_size, int(ds_size*cv_percent))

def create_val_set(idx):
    # return labels[labels==idx].copy(deep=True)
    return labels.iloc[idx]

def remove_val_from_train(idx):
    labels.drop(labels.index[[idx]], inplace=True)


class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w))

        # h and w are swapped for landmarks because for images,
        # x and y axes are axis 1 and 0 respectively
        landmarks = landmarks * [new_w / w, new_h / h]

        return {'image': img, 'landmarks': landmarks}


class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                      left: left + new_w]

        landmarks = landmarks - [left, top]

        return {'image': image, 'landmarks': landmarks}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'landmarks': torch.from_numpy(landmarks)}
