import mastplotlib.pyplot as plt
import numpy as np



def find_lr(learner, lr_init=1e-6, increase_factor=10):
    loss = []

    
