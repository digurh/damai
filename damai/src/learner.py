import numpy as np
import matplotlib.pyplot as plt

from torch import optim


class Learner:
    def __init__(self, data, net, opt=None):
        self.data = data
        self.net = net

        if opt is None: self.opt = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
        else self.opt = opt

    def lr_find(self, lr_init=1e-7, lr_end = 10, increase_factor=10):
        losses = self.train(lr_find=True)
        learn_rates = [lr_init**i for i in range()]
        for param_group in self.opt.param_groups:
            param_group['lr'] = lr


    def save(self):
        pass


    def load(self):
        pass


    def train(self, lr_find=False):
        pass
