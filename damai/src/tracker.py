import numpy as np
import matplotlib.pyplot as plt
from torch import optim


class Tracker:
    self.losses = []
    self.val_losses = []
    self.n_epochs = [1]
    self.batch_num = 0
    self.restart_sch = [1, 1, 1]

    def step(self): pass
    def plot(self): pass


class LRScheduler(Tracker):
    # restart_sch is (n_cycles, increase_factor_per_cycle)
    def __init__(self, opt=None, decay_type=None, learn_rate=0.0001):
        super().__init__()
        self.learn_rate = learn_rate

        self.opt = None
        self.decay_type = None
        if opt is None:
            self.opt = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
        else:
            self.opt = opt
        if decay_type is None:
            self.decay_type = optim.lr_scheduler.CosineAnnealingLR(self.opt, )
        else:
            self.decay_type = decay_type

    def step(self, lr_max):
        lr = 0.5 * lr_max * (1 + np.cos((T_curr / T_max) * np.pi))
        for param_group in opt.param_groups:
            param_group['lr'] = lr

    def set_decay(self, decay_type):
        self.decay_type = decay_type

    def set_restarts(self, sch):
        '''
        Allows a setting of the restart schedule to something other than the default

        Params: sch: [x, y, z] x = number of cycles, y = cycle length, z = cycle increase factor
        '''
        self.n_epochs = [sch[1] * sch[2]**i for i in range(sch[0])]
        self.restart_sch = sch


    def get_opt(self):
        return self.opt

    def set_opt(self, new_opt):
        self.opt = new_opt

    def get_sch(self):
        print("Restart Schedule: {} - Resulting Epochs: {}".format(self.restart_sch, self.n_epochs))


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

    def plot(self):

        fig = plt.figure()
        plt.plot(self.learn_rates, self.losses)
        plt.ylabel('loss')
        plt.xlabel('learn rates')
        plt.show()
