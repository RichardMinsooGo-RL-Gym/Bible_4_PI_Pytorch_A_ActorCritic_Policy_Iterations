import sys
IN_COLAB = "google.colab" in sys.modules

import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

import torch.multiprocessing as mp

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

class Worker(mp.Process):

    def __init__(self, id, env, gamma, global_network, global_optimizer, global_episode, GLOBAL_MAX_EPISODE):
        super(Worker, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.name = "w%i" % id
        
        self.env = env
        
        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.n
        
        self.gamma = gamma
        
        self.local_network = TwoHeadNetwork(self.state_size, self.action_size) 
        
        self.global_network = global_network
        self.global_episode = global_episode
        self.global_optimizer = global_optimizer
        self.GLOBAL_MAX_EPISODE = GLOBAL_MAX_EPISODE

        # sync local networks with global networks
        self.sync_with_global()
    
    def get_action(self, state):
        state = torch.FloatTensor(state).to(self.device)
        logits, _ = self.local_network.forward(state)
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
        logits, values = self.local_network.forward(states)
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
    
    def update_global(self, trajectory):
        loss = self.compute_loss(trajectory)
        
        self.global_optimizer.zero_grad()
        loss.backward()
        # propagate local gradients to global parameters
        for local_params, global_params in zip(self.local_network.parameters(), self.global_network.parameters()):
            global_params._grad = local_params._grad
        self.global_optimizer.step()

    def sync_with_global(self):
        self.local_network.load_state_dict(self.global_network.state_dict())

    def run(self):

        for self.global_episode.value in range(self.GLOBAL_MAX_EPISODE):
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
                    with self.global_episode.get_lock():
                        self.global_episode.value += 1
            if (self.global_episode.value+1) % 10 == 0:
                print("Episode " + str(self.global_episode.value+1) + ": " + str(episode_reward))
            
            self.update_global(trajectory)
            self.sync_with_global()
        
class A3CAgent:
    
    def __init__(self, env, gamma, lr, global_max_episode):
        self.env = env
        
        self.gamma = gamma
        self.lr = lr
        self.global_episode = mp.Value('i', 0)
        self.GLOBAL_MAX_EPISODE = global_max_episode

        self.global_network = TwoHeadNetwork(self.env.observation_space.shape[0], self.env.action_space.n)
        self.global_network.share_memory()
        self.global_optimizer = optim.Adam(self.global_network.parameters(), lr=lr) 
        self.workers = [Worker(i, env, self.gamma, self.global_network, self.global_optimizer, self.global_episode, self.GLOBAL_MAX_EPISODE) for i in range(mp.cpu_count())]
    
    def train(self):
        print("Training on {} cores".format(mp.cpu_count()))
        input("Enter to start")

        [worker.start() for worker in self.workers]
        [worker.join() for worker in self.workers]
    
    def save_model(self):
        torch.save(self.global_network.state_dict(), "a3c_model.pth")

if __name__ == "__main__":
    
    env_name = "CartPole-v0"
    # set environment
    env.seed(1)     # reproducible, general Policy gradient has high variance
    lr = 1e-4
    gamma = 0.99
    GLOBAL_MAX_EPISODE = 1000  # Set total number of episodes to train agent on.

    env = gym.make(env_name)
    agent = A3CAgent(env, gamma, lr, GLOBAL_MAX_EPISODE)
    agent.train()
    agent.save_model()


