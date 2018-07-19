import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from torch import optim


class Learner:
    def __init__(self, data, net):
        '''
        Encapsulates entire learning 'process' into one object

        Params: data: instance of DataHandler to be assiciated with network
                net: instance of Network(nn.Module) to be trained
        '''
        self.data = data
        self.net = net

    def lr_find(self, lr_init=1e-7, lr_end = 1, increase_factor=10):
        '''
        Allows optimal learning rate to be found using the method from Cyclical
        Learning Rates for Training Neural Networks: learning rate starts very
        low and slowly raises until loss stops decreasing. Implementation in
        LRFinder()

        Params: lr_init: starting value from which lr will be increased
                lr_end: lr will increase until it is >= this value
                increase_factor: amount by which previous learning rate will be
                                 multiplied each step
        '''
        losses = self.train(sch=LRFinder(lr_init, lr_end, increase_factor))


    def save(self):
        pass


    def load(self):
        pass


    def train(self, sch):

        for i, (X, y) in tqdm(enumerate(self.data.batch_load())):

            sch.step()



    def run_val_set(self):

        return acc

    def print_log(self):
