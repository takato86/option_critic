import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

if __name__ == "__main__":
    step_list = []
    avgduration_list = []
    history = np.load("optioncritic-fourrooms-baseline_True-discount_0.99-epsilon_0.01-lr_critic_0.5-lr_intra_0.25-lr_term_0.25-nepisodes_250-noptions_4-nruns_50-nsteps_1000-primitive_False-temperature_0.01.npy")
    for run in history:
        for episode in run:
            step_list.append(episode[0])
            avgduration_list.append(episode[1])
        
    step_list = np.array(step_list)
    avgduration_list = np.array(avgduration_list)
    step_list = step_list[:250]
    avgduration_list = avgduration_list[:250]

    num_episodes = range(len(step_list))

    plt.subplot(2,1,1)
    plt.plot(num_episodes, step_list)
    plt.title("step_series")
    
    plt.subplot(2,1,2)
    plt.plot(num_episodes, avgduration_list)
    plt.title("avgdur_series")

    plt.tight_layout()
    plt.show()
    

