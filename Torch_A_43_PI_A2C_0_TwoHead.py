import sys
IN_COLAB = "google.colab" in sys.modules

import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical


class TwoHeadNetwork(nn.Module):

    def __init__(self, state_size, action_size):
        super(TwoHeadNetwork, self).__init__()
        
        self.state_size = state_size
        self.action_size = action_size
        
        self.policy1 = nn.Linear(self.state_size, 256) 
        self.policy2 = nn.Linear(256, self.action_size)

        self.value1 = nn.Linear(self.state_size, 256)
        self.value2 = nn.Linear(256, 1)
        
    def forward(self, state):
        logits = F.relu(self.policy1(state))
        poicy = self.policy2(logits)

        value = F.relu(self.value1(state))
        value = self.value2(value)

        return poicy, value

class A2CAgent():

    def __init__(self, env, gamma, lr):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.env = env
        
        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.n
        
        self.gamma = gamma
        self.lr = lr
        
        self.model = TwoHeadNetwork(self.state_size, self.action_size)
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
    
    def get_action(self, state):
        state = torch.FloatTensor(state).to(self.device)
        logits, _ = self.model.forward(state)
        dist = F.softmax(logits, dim=0)
        probs = Categorical(dist)

        return probs.sample().cpu().detach().item()
    
    def compute_loss(self, trajectory):
        states      = torch.FloatTensor([sars[0] for sars in trajectory]).to(self.device)
        actions     = torch.LongTensor([sars[1] for sars in trajectory]).view(-1, 1).to(self.device)
        rewards     = torch.FloatTensor([sars[2] for sars in trajectory]).to(self.device)
        next_states = torch.FloatTensor([sars[3] for sars in trajectory]).to(self.device)
        dones       = torch.FloatTensor([sars[4] for sars in trajectory]).view(-1, 1).to(self.device)
        
        # compute discounted rewards
        discounted_rewards = [torch.sum(torch.FloatTensor([self.gamma**i for i in range(rewards[j:].size(0))])\
             * rewards[j:]) for j in range(rewards.size(0))]  # sorry, not the most readable code.
        value_targets = rewards.view(-1, 1) + torch.FloatTensor(discounted_rewards).view(-1, 1).to(self.device)
        
        # compute policy loss with entropy bonus
        logits, values = self.model.forward(states)
        dists = F.softmax(logits, dim=1)
        probs = Categorical(dists)
        
        # compute value loss
        critic_loss = F.mse_loss(values, value_targets.detach())
        
        # compute entropy bonus
        entropy = []
        for dist in dists:
            entropy.append(-torch.sum(dist.mean() * torch.log(dist)))
        entropy = torch.stack(entropy).sum()
        
        # compute policy loss
        advantage = value_targets - values
        actor_loss = -probs.log_prob(actions.view(actions.size(0))).view(-1, 1) * advantage.detach()
        actor_loss = actor_loss.mean() - 0.001 * entropy
        
        loss = actor_loss + critic_loss 
        return loss
    
    def train_step(self, trajectory):
        loss = self.compute_loss(trajectory)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


    def train(self, max_episodes=1000):

        episode = 0
        for episode in range(max_episodes):
            episode_reward = 0
            done = False
            state = self.env.reset()
            trajectory = [] # [[s, a, r, s', done], [], ...]

            
            while not done:
                # env.render()
                action = self.get_action(state)

                # print(action)
                next_state, reward, done, _ = self.env.step(action)
                
                trajectory.append([state, action, reward, next_state, done])

                state = next_state
                episode_reward += reward

                if done:
                    # state, action, reward, next_state, done
                    break
            if (episode+1) % 10 == 0:
                print("Episode " + str(episode+1) + ": " + str(episode_reward))
            self.train_step(trajectory)

if __name__ == "__main__":
    
    env_name = "CartPole-v0"
    # set environment
    env.seed(1)     # reproducible, general Policy gradient has high variance
    lr = 1e-4
    gamma = 0.99
    hidden_size = 64
    max_episodes = 500  # Set total number of episodes to train agent on.
    env = gym.make(env_name)
    agent = A2CAgent(env, gamma, lr)
    agent.train(max_episodes)


