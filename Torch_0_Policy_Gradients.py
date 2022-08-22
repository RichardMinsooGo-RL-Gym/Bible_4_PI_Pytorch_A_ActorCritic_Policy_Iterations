from itertools import count

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

gamma  = 0.99
seed   = 1234
render = False
log_interval = 10

env_name = "CartPole-v0"
# set environment
env = gym.make(env_name)
env.seed(seed)     # reproducible, general Policy gradient has high variance

state_size  = env.observation_space.shape[0]
action_size = env.action_space.n
hidden_size = 128
max_episodes = 2000  # Set total number of episodes to train agent on.

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.affine1 = nn.Linear(state_size, hidden_size)
        self.dropout = nn.Dropout(p=0.6)
        self.action_head = nn.Linear(hidden_size, action_size)

        self.saved_log_probs = []
        self.rewards = []

    def forward(self, state):
        x = self.affine1(state)
        x = self.dropout(x)
        x = F.relu(x)
        action_prob = self.action_head(x)
        return F.softmax(action_prob, dim=1)


model = Network()
optimizer = optim.Adam(model.parameters(), lr=3e-2)
eps = np.finfo(np.float32).eps.item()


def get_action(state):
    state = torch.from_numpy(state).float().unsqueeze(0)
    probs = model(state)

    # create a categorical distribution over the list of probabilities of actions
    m = Categorical(probs)

    # and sample an action using the distribution
    action = m.sample()

    # save to action buffer
    model.saved_log_probs.append(m.log_prob(action))

    # the action to take (left or right)
    return action.item()


def train_step():
    R = 0
    saved_log_probs = model.saved_log_probs
    policy_losses = [] # list to save actor (policy) loss

    returns = [] # list to save the true values

    # calculate the true value using rewards returned from the environment
    for r in model.rewards[::-1]:
        # calculate the discounted value
        R = r + gamma * R
        returns.insert(0, R)
    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + eps)

    for log_prob, R in zip(saved_log_probs, returns):
        policy_losses.append(-log_prob * R)

    # reset gradients
    optimizer.zero_grad()
    loss = torch.cat(policy_losses).sum()

    # perform backprop
    loss.backward()
    optimizer.step()

    # reset rewards and action buffer
    del model.rewards[:]
    del model.saved_log_probs[:]


if __name__ == "__main__":
    running_reward = 10

    # run inifinitely many episodes
    for episode in range(max_episodes):

        # reset environment and episode reward
        state = env.reset()
        episode_reward = 0

        # for each episode, only run 9999 steps so that we don't 
        # infinite loop while learning
        for t in range(1, 200):

            # select action from policy
            action = get_action(state)

            # take the action
            next_state, reward, done, _ = env.step(action)


            model.rewards.append(reward)
            episode_reward += reward
            if done:
                break

            state = next_state
            
        # update cumulative reward
        running_reward = 0.05 * episode_reward + (1 - 0.05) * running_reward

        # perform backprop
        train_step()

        # log results
        if (episode+1) % log_interval == 0:
            print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
                  episode+1, episode_reward, running_reward))

        # check if we have "solved" the cart pole problem
        if running_reward > env.spec.reward_threshold:
            print("Solved! Running reward is now {} and "
                  "the last episode runs to {} time steps!".format(running_reward, t))
            break

