import numpy as np
from operator import itemgetter

class ReplayBuffer(object):
    '''
    s   : current obs
    omega: current option
    a   : taken action
    r   : reward
    s_  : next obs
    t   : is termination
    ''' 
    def __init__(self):
        self.history = []
    
    def add_experience(self, s, omega, a, r, s_, t):
        experience = (s,omega, a,r,s_,t)
        self.history.append(experience)
    
    def get_experience(self, size):
        shuffled_idx = np.random.choice(np.arange(len(self.history)), size, replace=False)
        return itemgetter(*shuffled_idx)(self.history)
    