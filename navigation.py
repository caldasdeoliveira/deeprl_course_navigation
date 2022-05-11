# # Training script
#
# ## Imports

# +
from unityagents import UnityEnvironment
import numpy as np
from collections import namedtuple, deque
import torch

import matplotlib.pyplot as plt
# %matplotlib inline
# -

# ## Initializing environment

env = UnityEnvironment(file_name="Banana.app")

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# Value for random seed
seed = 42

# ### Checking environment and setting utility variables

# +
# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# number of agents in the environment
print('Number of agents:', len(env_info.agents))

# number of actions
action_size = brain.vector_action_space_size
print('Number of actions:', action_size)

# examine the state space 
state = env_info.vector_observations[0]
print('States look like:', state)
state_size = len(state)
print('States have length:', state_size)
# -

# ## Instantiate agent

# +
from dqn import Agent

#instantiate agent
agent = Agent(state_size, action_size, seed)


# -

# ## Training

# +
def train_dqn_agent(n_episodes=2000, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
    """ Train the instantiated agent and saves the best model.

        Args:
            n_episodes (int): number of episodes to train on.
            max_t (int): maximum number of actions an episode contains
            eps_start (int): starting value for epsilon
            eps_end (int): minimum value that epsilon can take
            eps_decay (int): epsilon decay factor per episode
            
        Returns:
            scores (list[int]): list with the score in each episode

        """
    
    scores = []                      # list containing scores from each episode
    scores_window = deque(maxlen=100)# last 100 scores
    eps = eps_start                  # initialize epsilon
    best_score_yet = 13.0            # best score so far, determines when to save
    
    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode = True)[brain_name] # reset the environment
        state = env_info.vector_observations[0]            # get the current state
        score = 0
        for t in range(max_t):
            action = agent.act(state, eps)
            
            # update env
            env_info = env.step(action)[brain_name]        # send the action to the environment
            next_state = env_info.vector_observations[0]   # get the next state
            reward = env_info.rewards[0]                   # get the reward
            done = env_info.local_done[0]                  # see if episode has finished
            
            # provide agent with the results
            agent.step(state, action, reward, next_state, done)
            
            state = next_state
            score += reward
            
            if done:
                break
        scores_window.append(score)       # save most recent score
        scores.append(score)              # save most recent score
        eps = max(eps_end, eps_decay*eps) # decrease epsilon
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        if np.mean(scores_window) >= best_score_yet:
            if best_score_yet == 13.0:
                print(
                    f'\nEnvironment solved in {i_episode-100:d} episodes!'\
                    f'\tAverage Score: {np.mean(scores_window):.2f}'
                )
            else:
                print(f"\nSaving better agent with Average Score: {np.mean(scores_window):.2f}")
            torch.save(agent.qnetwork_local.state_dict(), 'checkpoint.pth')
            best_score_yet += 1
    return scores

scores = train_dqn_agent()
# -

# plot the scores
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(len(scores)), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()

# ## Clean up

env.close()


