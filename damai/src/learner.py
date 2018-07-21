import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from torch import optim


class Learner:
    def __init__(self, data, net, criterion):
        '''
        Encapsulates entire learning 'process' into one object

        Params: data: instance of DataHandler to be assiciated with network
                net: instance of Network(nn.Module) to be trained
        '''
        self.data = data
        self.net = net
        self.criterion = criterion

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
        losses = self.train(sch=LRFinder(lr_init, lr_end, increase_factor), learn=False)

    def fit(self, sch):
        for i in range(len(sch.n_epochs)):
            self.train(sch, sch.n_epochs[i])

        # self.save()

    def train(self, sch, n_epochs, learn=True):
        for ep in range(n_epochs):
            ep_losses = []
            for i, (X, y) in tqdm(enumerate(self.data.batch_load())):
                y_hat = self.net.forward(X)
                loss = self.criterion(y_hat, y)
                sch.losses.append(loss)
                ep_losses.append(loss)

                # sch.step()
                sch.decay_type.step()

                sch.opt.zero_grad()
                self.net.backward(retain_graph=True)
                sch.opt.step()

                sch.batch_num += 1

            if learn:
                val_loss = self.run_val_set()
                sch.val_losses.append(val_loss)
                self.print_log(ep, ep_losses, val_losses)
            else:
                sch.plot()


    def run_val_set(self):
        val_losses = []
        for i, (X, y) in enumerate(self.data.val_load()):
            y_hat = self.net.forward(X)
            loss = self.criterion(y_hat, y)
            val_losses.append(loss)

        return val_losses

    def print_log(self, ep, train_loss, val_loss):
        print('Epoch: {} - Train Loss: {:.3f} - {}\Val Loss: {:.3f}'.format(ep, np.mean(train_loss), np.mean(val_loss)))

    def save(self):
        pass

    def load(self):
        pass
