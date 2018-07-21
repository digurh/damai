import numpy as np
import matplotlib.pyplot as plt
from torch import optim


class Tracker:
    self.losses = []
    self.val_losses = []
    self.n_epochs = 0
    self.batch_num = 0
    self.restart_sch = (1, 1, 1)

    def step(self): pass
    def plot(self): pass


class LRScheduler(Tracker):
    # restart_sch is (n_cycles, increase_factor_per_cycle)
    def __init__(self, opt=None, learn_rate=0.0001):
        super().__init__()
        self.learn_rate = learn_rate
        self.decay_type = None
        if opt is None: self.opt = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
        else self.opt = opt

    def step(self):
        self.decay_type.step()

    def set_decay(self, decay_type):
        self.decay_type = decay_type

    def set_restarts(self, sch):
        '''
        Allows a setting of the restart schedule to something other than the default

        Params: sch: (x, y, z) x = number of cycles, y = cycle length, z = cycle increase factor
        '''
        self.restart_sch = sch
        self.n_epochs = sum([sch[1] * sch[2]**i for i in range(sch[0])])

    def get_opt(self):
        return self.opt

    def set_opt(self, new_opt):
        self.opt = new_opt

    def get_sch(self):
        return self.restart_sch


    def plot(self):
        fig = plt.figure()
        plt.plot(np.arange(1, self.batch_num), self.losses)
        plt.ylabel('loss')
        plt.xlabel('batch')
        plt.show()


class LRFinder(Tracker):
    def __init__(self, lr_init=1e-7, lr_end = 1, increase_factor=10):
        super().__init__()
        self.lr_init = lr_init
        self.lr_end = lr_end
        self.increase_factor = increase_factor

        self.learn_rates = [lr_init]
        while learn_rates[-1] < lr_end:
            self.learn_rates.append(learn_rates[-1]*increase_factor)

    def step(self):
        for param_group in self.opt.param_groups:
            param_group['lr'] *= self.increase_factor

        self.batch_num += 1

    def plot(self):

        fig = plt.figure()
        plt.plot(self.learn_rates, self.losses)
        plt.ylabel('loss')
        plt.xlabel('learn rates')
        plt.show()
