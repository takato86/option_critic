import numpy as np

class Experience_Replay(object):
    '''
    s   : current obs
    a   : taken action
    r   : reward
    s_  : next obs
    t   : is termination
    ''' 
    def __init___(self):
        self.history = []
    
    def add_experience(self, experience):
        self.history.append(experience)
    
    def get_experience(self, size):
        experience = np.array(self.history)
        return np.random.choice(experience, size)
    