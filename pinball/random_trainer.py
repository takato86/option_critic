import numpy as np

class RandomTrainer(object):
    def __init__(self):
        pass

    def teach(self):
        rand = np.random.rand()
        if rand < 0.1:
            return True
        else:
            return False
