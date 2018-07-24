import torch
from torch import Dataset, DataLoader


class DataHandler:
    def __init__(self, batch_size, image_size=None):


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
