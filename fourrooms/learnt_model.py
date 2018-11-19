import gym
import argparse
import numpy as np
from fourrooms import Fourrooms

from scipy.special import expit
from scipy.misc import logsumexp
import dill
import os

import seaborn as sns
import matplotlib.pyplot as plt

class Tabular:
    def __init__(self, nstates):
        self.nstates = nstates

    def __call__(self, state):
        return np.array([state,])

    def __len__(self):
        return self.nstates

if __name__ == "__main__":
    rng = np.random.RandomState(1234)
    env = gym.make('Fourrooms-v0')
    features = Tabular(env.observation_space.n)
    nfeatures, nactions = len(features), env.action_space.n
    nepisodes = 1
    nsteps = 1000
    possible_next_goals = [68, 69, 70, 71, 72, 78, 79, 80, 81, 82, 88, 89, 90, 91, 92, 93, 99, 100, 101, 102, 103]
    rng = np.random.RandomState(1234)

    with open('oc-options.pl', 'rb') as f:
        obj = dill.loads(f.read())
    
    intra_policies = obj['intra_policies']
    policy = obj['policy']
    term_list = obj['term']
    
    # only one run
    for episode in range(nepisodes):
        if episode==1000:
            env.goal = rng.choice(possible_next_goals)
            print('************* New goal : ', env.goal)
        obs = env.reset()
        phi = features(obs)
        start = env.unwrapped.to_cell(obs)
        goal = env.unwrapped.to_cell(env.unwrapped.goal)
        print("start:{}".format(start))
        print("goal:{}".format(goal))
        option = policy.sample(phi)
        action = intra_policies[option].sample(phi)
        cumreward = 0.
        duration = 1
        option_switches = 0
        avgduration = 0.
        for step in range(nsteps):
            observation, reward, done, _ = env.step(action)
            phi = features(observation)
            if term_list[option].sample(phi):
                option = policy.sample(phi)
                option_switches += 1
                avgduration += (1./option_switches)*(duration - avgduration)
                duration = 1
            
            action = intra_policies[option].sample(phi)
            cumreward += reward
            duration += 1
            if done:
                break

        print("num_steps:{}".format(step))
        print("cumreward:{}".format(cumreward))
        print("duration:{}".format(duration))
    
    # Show termination function distribution
    template = np.zeros
    term_class = term_list[1]
    env_shape = env.unwrapped.occupancy.shape

    term_prob = {}
    fig_prob = {}
    for i, term in enumerate(term_list):
        print('term-{}'.format(i))
        prob_list = []
        env_exp = np.full(env_shape, -1, dtype=np.float)
        for feat in range(nfeatures):
            prob_list.append(term.pmf(feat))
            env_exp[env.unwrapped.to_cell(feat)[1]][env.unwrapped.to_cell(feat)[0]] = term.pmf(feat)
        env_exp[goal[1]][goal[0]] = 0.8
        term_prob[str(i)] = prob_list
        fig_prob[str(i)] = env_exp
        #print("{}th termination func: {} ".format(i,prob_list))

    plt.subplot(2,2,1)
    sns.heatmap(fig_prob["0"])
    plt.subplot(2,2,2)
    sns.heatmap(fig_prob["1"])
    plt.subplot(2,2,3)
    sns.heatmap(fig_prob["2"])
    plt.subplot(2,2,4)
    sns.heatmap(fig_prob["3"])
    plt.show()
    
# oc-options.pl
# optioncritic-fourrooms-baseline_True-discount_0.99-epsilon_0.01-lr_critic_0.5-lr_intra_0.25-lr_term_0.25-nepisodes_250-noptions_4-nruns_50-nsteps_1000-primitive_False-temperature_0.01.npy