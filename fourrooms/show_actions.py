import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import gym
import dill
from transfer import Tabular
import seaborn as sns
from scipy.misc import logsumexp

if __name__=='__main__':
    rng = np.random.RandomState(1234)
    env = gym.make('Fourrooms-v0')
    features = Tabular(env.observation_space.n)
    nfeatures, nactions = len(features), env.action_space.n
    nepisodes = 1
    nsteps = 1000
    possible_next_goals = [68, 69, 70, 71, 72, 78, 79, 80, 81, 82, 88, 89, 90, 91, 92, 93, 99, 100, 101, 102, 103]
    rng = np.random.RandomState(1234)
    env_shape = env.unwrapped.occupancy.shape

    with open('models/oc-options.pl', 'rb') as f:
        obj = dill.loads(f.read())
    
    intra_policies = obj['intra_policies']
    policy = obj['policy']
    term_list = obj['term']

    intra_policy_list = {}
    tmp_policy = []
    for i, intra_policy in enumerate(intra_policies):
        exp_tmp = np.full(env_shape, -1)
        for feat in range(nfeatures):
            phi = features(feat)
            action = int(np.argmax(intra_policy.value(phi)))
            exp_tmp[env.unwrapped.to_cell(feat)[0]][env.unwrapped.to_cell(feat)[1]] = action
        intra_policy_list[str(i)] = exp_tmp
    
    policy_over_options = np.full(env_shape, -1)
    for feat in range(nfeatures):
        phi = features(feat)
        action = int(np.argmax(policy.value(phi)))
        policy_over_options[env.unwrapped.to_cell(feat)[0]][env.unwrapped.to_cell(feat)[1]] = action+1
    goal = env.unwrapped.goal
    policy_over_options[env.unwrapped.to_cell(goal)[0]][env.unwrapped.to_cell(goal)[1]] = 0
    plt.subplot(2,3,1)
    plt.title("Option 1")
    sns.heatmap(intra_policy_list["0"], annot=True)
    plt.subplot(2,3,2)
    plt.title("Option 2")
    sns.heatmap(intra_policy_list["1"], annot=True)
    plt.subplot(2,3,4)
    plt.title("Option 3")
    sns.heatmap(intra_policy_list["2"], annot=True)
    plt.subplot(2,3,5)
    plt.title("Option 4")
    sns.heatmap(intra_policy_list["3"], annot=True)
    plt.subplot(2,3,3)
    plt.title("Policy over Options")
    sns.heatmap(policy_over_options, annot = True)

    plt.tight_layout()
    plt.show()
    

            