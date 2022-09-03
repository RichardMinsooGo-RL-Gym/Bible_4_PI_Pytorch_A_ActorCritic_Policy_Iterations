import sys
IN_COLAB = "google.colab" in sys.modules

import gym
import torch
import torch.nn as nn
import numpy as np 
import torch.nn.functional as F
import torch.optim as optim
import random
import torch.autograd as autograd
from collections import deque

if torch.backends.cudnn.enabled:
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

seed = 777
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

class ReplayBuffer:
    
    def __init__(self, max_size):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)

    def store(self, state, action, reward, next_state, done):
        experience = (state, action, np.array([reward]), next_state, done)
        self.buffer.append(experience)

    def sample_batch(self, batch_size):
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []
        
        batch = random.sample(self.buffer, batch_size)

        for experience in batch:
            state, action, reward, next_state, done = experience
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.buffer)

# Ornstein-Ulhenbeck Noise
# Taken from #https://github.com/vitchyr/rlkit/blob/master/rlkit/exploration_strategies/ou_strategy.py
class OUNoise(object):
    def __init__(self, action_space, mu=0.0, theta=0.15, max_sigma=0.3, min_sigma=0.3, decay_period=100000):
        self.mu           = mu
        self.theta        = theta
        self.sigma        = max_sigma
        self.max_sigma    = max_sigma
        self.min_sigma    = min_sigma
        self.decay_period = decay_period
        self.action_size  = action_space.shape[0]
        self.low          = action_space.low
        self.high         = action_space.high
        self.reset()
        
    def reset(self):
        self.state = np.ones(self.action_size) * self.mu
        
    def evolve_state(self):
        x  = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_size)
        self.state = x + dx
        return self.state
    
    def get_action(self, action, t=0):
        ou_state = self.evolve_state()
        self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(1.0, t / self.decay_period)
        return np.clip(action + ou_state, self.low, self.high)

class Actor(nn.Module):
    
    def __init__(self, state_size, action_size):
        super(Actor, self).__init__()
        
        self.state_size = state_size
        self.action_size = action_size
        
        self.fc1 = nn.Linear(self.state_size, 512)
        self.linear2 = nn.Linear(512, 128)
        self.fc_out = nn.Linear(128, self.action_size)
    
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.linear2(x))
        x = torch.tanh(self.fc_out(x))

        return x

class CriticQ(nn.Module):
    
    def __init__(self, state_size, action_size):
        super(CriticQ, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        
        self.fc1 = nn.Linear(self.state_size, 1024)
        self.linear2 = nn.Linear(1024 + self.action_size, 512)
        self.linear3 = nn.Linear(512, 300)
        self.fc_out = nn.Linear(300, 1)

    def forward(self, state, action):
        x = F.relu(self.fc1(state))
        xa_cat = torch.cat([x,action], 1)
        xa = F.relu(self.linear2(xa_cat))
        xa = F.relu(self.linear3(xa))
        q_value = self.fc_out(xa)

        return q_value

class TD3Agent:
    
    def __init__(self, env, gamma, tau, buffer_maxlen, delay_step, noise_std, noise_bound, critic_lr, actor_lr):
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.env = env
        
        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.shape[0]
        
        # hyperparameters
        self.gamma = gamma
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.tau   = tau
        self.noise_std = noise_std
        self.noise_bound = noise_bound
        self.update_step = 0 
        self.delay_step = delay_step
        
        # initialize networks 
        self.actor = Actor(self.state_size, self.action_size).to(self.device)
        self.actor_target = Actor(self.state_size, self.action_size).to(self.device)
        
        self.critic1 = CriticQ(self.state_size, self.action_size).to(self.device)
        self.critic2 = CriticQ(self.state_size, self.action_size).to(self.device)
        self.critic1_target = CriticQ(self.state_size, self.action_size).to(self.device)
        self.critic2_target = CriticQ(self.state_size, self.action_size).to(self.device)
        
        # Copy critic target parameters
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)
        
        for target_param, param in zip(self.critic1_target.parameters(), self.critic1.parameters()):
            target_param.data.copy_(param.data)
        
        for target_param, param in zip(self.critic2_target.parameters(), self.critic2.parameters()):
            target_param.data.copy_(param.data)

        # initialize optimizers 
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=self.critic_lr)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=self.critic_lr)
        
        self.replay_buffer = ReplayBuffer(buffer_maxlen)
        
    def get_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        logits = self.actor.forward(state)
        action = logits.squeeze(0).cpu().detach().numpy()

        return action
    
    def train_step(self, batch_size):
        # states, actions, rewards, next_states, _ = self.replay_buffer.sample(batch_size)
        states, actions, rewards, next_states, dones = self.replay_buffer.sample_batch(batch_size)
        states      = torch.FloatTensor(states).to(self.device)
        actions     = torch.FloatTensor(actions).to(self.device)
        rewards     = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones       = torch.FloatTensor(dones).to(self.device)
        
        action_space_noise = self.generate_action_space_noise(actions)
        curr_Q1s = self.critic1.forward(states, actions)
        curr_Q2s = self.critic2.forward(states, actions)
        next_P_targs  = self.actor_target.forward(next_states) + action_space_noise
        next_Q1_targs = self.critic1_target.forward(next_states, next_P_targs)
        next_Q2_targs = self.critic2_target.forward(next_states, next_P_targs)
        expected_Qs   = rewards + self.gamma * torch.min(next_Q1_targs, next_Q2_targs)
        
        # critic loss
        critic1_loss = F.mse_loss(curr_Q1s, expected_Qs.detach())
        critic2_loss = F.mse_loss(curr_Q2s, expected_Qs.detach())
        
        # update critics
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()
        
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()
        
        # delyaed update for actor & target networks  
        if self.update_step % self.delay_step == 0:
            # actor
            actor_loss = -self.critic1(states, self.actor(states)).mean()
            
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            
            # target networks
            self.update_targets()

        self.update_step += 1

    def generate_action_space_noise(self, actions):
        noise = torch.normal(torch.zeros(actions.size()), self.noise_std).clamp(-self.noise_bound, self.noise_bound).to(self.device)
        return noise
    
    def update_targets(self):
        for target_param, param in zip(self.critic1_target.parameters(), self.critic1.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))
            
        for target_param, param in zip(self.critic2_target.parameters(), self.critic2.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))
            
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))


        # --------------------------------
    def train(self):

        episode_rewards = []
        
        for episode in range(max_episodes):
            episode_reward = 0
            done = False
            state = self.env.reset()

            for step in range(max_steps):
                action = self.get_action(state)
                
                next_state, reward, done, _ = self.env.step(action)
                
                self.replay_buffer.store(state, action, reward, next_state, done)
                
                state = next_state
                episode_reward += reward

                if len(self.replay_buffer) > batch_size:
                    self.train_step(batch_size)   

                if done or step == max_steps-1:
                    episode_rewards.append(episode_reward)
                    print("Episode " + str(episode+1) + ": " + str(episode_reward))
                    break

    def save_model(self):
        torch.save(self.critic1.state_dict(), "TD3_value1_model.pth")
        torch.save(self.critic2.state_dict(), "TD3_value2_model.pth")
        torch.save(self.actor.state_dict(), "DDPG_policy_model.pth")

if __name__ == "__main__":
    
    # env = gym.make("Pendulum-v0")
    # actor_lr = 1e-3
    # critic_lr = 1e-3
    
    env_name = "Pendulum-v1"
    # set environment
    env = gym.make(env_name)
    env.seed(1)     # reproducible, general Policy gradient has high variance
    actor_lr = 0.0005
    critic_lr = 0.001
    gamma = 0.99
    tau = 1e-2
    noise_std = 0.2
    bound = 0.5
    delay_step = 2
    buffer_maxlen = 100000
    
    hidden_size = 256
    max_episodes = 200
    max_steps = 500
    batch_size = 32
    
    agent = TD3Agent(env, gamma, tau, buffer_maxlen, delay_step, noise_std, bound, critic_lr, actor_lr)
    
    agent.train()
    agent.save_model()
    
