import pandas as pd
import numpy as np
import os
from skimage import io, transform

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils


# split off validation set
def get_val_idx(ds_size, cv_percent=0.2):
    return np.random.choice(ds_size, int(ds_size*cv_percent))

def create_val_set(idx):
    # return labels[labels==idx].copy(deep=True)
    return labels.iloc[idx]

def remove_val_from_train(idx):
    labels.drop(labels.index[[idx]], inplace=True)


class CustomDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data_frame = pd.read_csv(csv_file)
        self.root_dir = os.path.dirname(os.path.realpath(__file__))
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir,
                                self.data_frame.iloc[idx, 0])
        image = io.imread(img_name)
        landmarks = self.data_frame.iloc[idx, 1:].as_matrix()
        landmarks = landmarks.astype('float').reshape(-1, 2)
        sample = {'image': image}

        if self.transform:
            sample = self.transform(sample)

        return sample

class DataHandler:
    def __init__(self, batch_size, image_size=None):
        self.batch_size = batch_size
        self.image_size = image_size
        self.loader = None
        self.transform = transforms.Compose([transforms.resize((self.image_size, self.image_size))
                                             transforms.RandomResizedCrop(),
                                             transforms.RandomHorizontalFlip(),
                                             transforms.ToTensor(),
                                             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                  std=[0.229, 0.224, 0.225])
                                            ])


    def get_csv(self, path, p_val=0.2, p_test = 0.2):
    '''
        Takes path to csv file and breaks file down into appropriate training,
        validation, and test sections that can be accessed via instance variabless
    '''
        dataset = CustomDataset(path, self.transform)
        self.loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)

    def get_data(self, path):
    '''
        Takes path to data folders and places training, validation, and test data
        in accessible instance variables
    '''
    train_set=torchvision.datasets.ImageFolder(root='/data/train/', transform=self.transform)
    val_set=torchvision.datasets.ImageFolder(root='/data/val/', transform=None)


    def batch_load(self):
    '''
        Generator function to load minibatches to network
    '''
        return self.loader
