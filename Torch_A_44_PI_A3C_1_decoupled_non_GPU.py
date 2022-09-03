import sys
IN_COLAB = "google.colab" in sys.modules

import gym
import torch
import torch.nn as nn
import numpy as np 
import torch.nn.functional as F
import torch.optim as optim
import random
from torch.distributions import Categorical

if torch.backends.cudnn.enabled:
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

seed = 777
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

import torch.multiprocessing as mp

GLOBAL_EP = 0

class Actor(nn.Module):
    
    def __init__(self, state_size, action_size):
        super(Actor, self).__init__()
        
        self.state_size = state_size
        self.action_size = action_size
        
        self.policy1 = nn.Linear(self.state_size, 256) 
        self.fc_out = nn.Linear(256, self.action_size)
    
    def forward(self, state):
        logits = F.relu(self.policy1(state))
        logits = self.fc_out(logits)

        return logits

class Critic(nn.Module):
    
    def __init__(self, state_size):
        super(Critic, self).__init__()
        self.state_size = state_size
        
        self.value1 = nn.Linear(self.state_size, 256)
        self.value2 = nn.Linear(256, 1)
        
    def forward(self, state):
        value = F.relu(self.value1(state))
        value = self.value2(value)

        return value

class Worker(mp.Process):
    
    def __init__(self, id, env, gamma, global_actor, global_critic, global_actor_optimizer, global_critic_optimizer):
        super(Worker, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.name = "w%i" % id
        
        self.env = env
        
        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.n
        
        self.gamma = gamma
        self.actor = Actor(self.state_size, self.action_size,
                           )
        self.critic = Critic(self.state_size)
        
        self.global_actor = global_actor
        self.global_critic = global_critic
        self.global_actor_optimizer = global_actor_optimizer
        self.global_critic_optimizer = global_critic_optimizer
        
        # sync local networks with global networks
        self.sync_with_global()
    
    def get_action(self, state):
        state = torch.FloatTensor(state).to(self.device)
        logits = self.actor.forward(state)
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
        expected_Qs = rewards.view(-1, 1) + torch.FloatTensor(discounted_rewards).view(-1, 1).to(self.device)
        
        # compute policy loss with entropy bonus
        curr_Ps = self.actor.forward(states)
        dists = F.softmax(curr_Ps, dim=1)
        probs = Categorical(dists)
        
        # compute value loss
        curr_Qs = self.critic.forward(states)
        critic_loss = F.mse_loss(curr_Qs, expected_Qs.detach())
        
        # compute entropy bonus
        entropy = []
        for dist in dists:
            entropy.append(-torch.sum(dist.mean() * torch.log(dist)))
        entropy = torch.stack(entropy).sum()
        
        # compute policy loss
        advantage = expected_Qs - curr_Qs
        actor_loss = -probs.log_prob(actions.view(actions.size(0))).view(-1, 1) * advantage.detach()
        actor_loss = actor_loss.mean() - 0.001 * entropy
        
        return critic_loss, actor_loss
    
    def update_global(self, trajectory):
        critic_loss, actor_loss = self.compute_loss(trajectory)
        
        self.global_critic_optimizer.zero_grad()
        actor_loss.backward()
        # propagate local gradients to global parameters
        for local_params, global_params in zip(self.actor.parameters(), self.global_actor.parameters()):
            global_params._grad = local_params._grad
        self.global_critic_optimizer.step()

        self.global_actor_optimizer.zero_grad()
        critic_loss.backward()
        # propagate local gradients to global parameters
        for local_params, global_params in zip(self.critic.parameters(), self.global_critic.parameters()):
            global_params._grad = local_params._grad
        self.global_actor_optimizer.step()
        # --------------------------------
    def sync_with_global(self):
        self.actor.load_state_dict(self.global_actor.state_dict())
        self.critic.load_state_dict(self.global_critic.state_dict())
    
    def run(self):
        global GLOBAL_EP
        
        while max_episodes > GLOBAL_EP:
            episode_reward = 0
            done = False
            state = self.env.reset()
            
            trajectory = [] # [[s, a, r, s', done], [], ...]

            while not done:
                action = self.get_action(state)
                
                next_state, reward, done, _ = self.env.step(action)
                
                trajectory.append([state, action, reward, next_state, done])
                
                state = next_state
                episode_reward += reward
            self.update_global(trajectory)
            self.sync_with_global()

            print(self.name + ' | EP{} EpisodeReward={}'.format(GLOBAL_EP+1, episode_reward))
            GLOBAL_EP += 1

class A3CAgent:
    
    def __init__(self, env, gamma, actor_lr, critic_lr):
        self.env = env
        self.gamma = gamma
        # self.global_episode = mp.Value('i', 0)
        self.state_size = env.observation_space.shape[0]
        self.action_size = env.action_space.n
        
        self.global_actor = Actor(self.state_size, self.action_size,
                                 )
        self.global_critic = Critic(self.state_size)
        self.global_actor.share_memory()
        self.global_critic.share_memory()
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.global_actor_optimizer = optim.Adam(self.global_critic.parameters(), lr=actor_lr) 
        self.global_critic_optimizer = optim.Adam(self.global_actor.parameters(), lr=critic_lr) 
        
        self.num_workers = mp.cpu_count()
        
    
    def train(self):
        print("Training on {} cores".format(self.num_workers))
        input("Enter to start")
        
        self.workers = []
        for i in range(self.num_workers):
            env = gym.make(env_name)
            self.workers.append(Worker(i, env, self.gamma, self.global_actor, self.global_critic,\
                                       self.global_actor_optimizer, self.global_critic_optimizer))
        
        for worker in self.workers:
            worker.start()

        for worker in self.workers:
            worker.join()
    
        # [worker.start() for worker in self.workers]
        # [worker.join() for worker in self.workers]
    
    def save_model(self):
        torch.save(self.global_critic.state_dict(), "a3c_value_model.pth")
        torch.save(self.global_actor.state_dict(), "a3c_policy_model.pth")

if __name__ == "__main__":
    
    env_name = "CartPole-v0"
    # set environment
    env = gym.make(env_name)
    env.seed(1)     # reproducible, general Policy gradient has high variance
    actor_lr = 0.0005
    critic_lr = 0.001
    gamma = 0.99
    
    hidden_size = 128
    max_episodes = 500  # Set total number of episodes to train agent on.
    agent = A3CAgent(env, gamma, actor_lr, critic_lr)
    agent.train()
    agent.save_model()
    
