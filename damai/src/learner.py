import numpy as np
import matplotlib.pyplot as plt

from torch import optim


class Learner:
    def __init__(self, data, net, opt=None):
        self.data = data
        self.net = net

        if opt is None: self.opt = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
        else self.opt = opt

    def lr_find(self, lr_init=1e-7, lr_end = 1, increase_factor=10):
        '''
        Allows optimal learning rate to be found using the method from Cyclical
        Learning Rates for Training Neural Networks: learnig rate starts very
        low and slowly raises until loss stops decreasing.

        Params: lr_init: starting value from which lr will be increased
                lr_end: lr will increase until it is >= this value
                increase_factor: amount by which previous learning rate will be 
                                 multiplied each step
        '''
        losses = self.train(lr_find=(lr_init, lr_end, increase_factor))
        learn_rates = [lr_init]

        while learn_rates[-1] < lr_end:
            learn_rates.append(learn_rates[-1]*increase_factor)

        self.lr_plot(learn_rates, losses)

    def __lr_find_update(self, i_f):
        for param_group in self.opt.param_groups:
            param_group['lr'] *= i_f

    def save(self):
        pass


    def load(self):
        pass


    def train(self, lr_find=None):
        pass


    def __lr_plot(self, *args):
        fig = plt.figure()
        plt.plot(args[0], args[1])
        plt.ylabel('loss')
        plt.xlabel('learn rate')
        plt.show()
