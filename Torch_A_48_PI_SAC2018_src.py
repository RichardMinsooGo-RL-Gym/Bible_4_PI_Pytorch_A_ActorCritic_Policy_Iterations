import sys
IN_COLAB = "google.colab" in sys.modules

from torch.distributions import Normal
import random
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
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

class Actor(nn.Module):
    
    def __init__(self, state_size, action_size, hidden_size=256, init_w=3e-3, log_std_min=-20, log_std_max=2):
        super(Actor, self).__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.linear1 = nn.Linear(state_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)

        self.mean_linear = nn.Linear(hidden_size, action_size)
        self.mean_linear.weight.data.uniform_(-init_w, init_w)
        self.mean_linear.bias.data.uniform_(-init_w, init_w)

        self.log_std_linear = nn.Linear(hidden_size, action_size)
        self.log_std_linear.weight.data.uniform_(-init_w, init_w)
        self.log_std_linear.bias.data.uniform_(-init_w, init_w)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))

        mean    = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

        return mean, log_std

    def sample(self, state, epsilon=1e-6):
        mean, log_std = self.forward(state)
        std = log_std.exp()

        normal = Normal(mean, std)
        z = normal.rsample()
        action = torch.tanh(z)

        log_pi = normal.log_prob(z) - torch.log(1 - action.pow(2) + epsilon)
        log_pi = log_pi.sum(1, keepdim=True)

        return action, log_pi

class CriticQ(nn.Module):
    
    def __init__(self, state_size, action_size, hidden_size=256, init_w=3e-3):
        super(CriticQ, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        
        self.linear1 = nn.Linear(self.state_size + self.action_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.fc_out = nn.Linear(hidden_size, 1)

        self.fc_out.weight.data.uniform_(-init_w, init_w)
        self.fc_out.bias.data.uniform_(-init_w, init_w)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        q_value = self.fc_out(x)
        return q_value

class CriticV(nn.Module):

    def __init__(self, input_dim, action_size, init_w=3e-3):
        super(CriticV, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, action_size)

        self.fc3.weight.data.uniform_(-init_w, init_w)
        self.fc3.bias.data.uniform_(-init_w, init_w)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        V_value = self.fc3(x)

        return V_value

class SACAgent:
    
    def __init__(self, env, gamma, tau, delay_step, v_lr, critic_lr, actor_lr, buffer_maxlen):
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.env = env
        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.shape[0]
        
        self.action_range = [env.action_space.low, env.action_space.high]
        
        # hyperparameters
        self.gamma = gamma
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.tau   = tau
        self.update_step = 0 
        self.delay_step = delay_step
        
        # initialize networks 
        self.actor = Actor(self.state_size, self.action_size).to(self.device)
        
        self.critic1 = CriticQ(self.state_size, self.action_size).to(self.device)
        self.critic2 = CriticQ(self.state_size, self.action_size).to(self.device)
        self.value_net = CriticV(self.state_size, self.action_size).to(self.device)
        self.target_value_net = CriticV(self.state_size, self.action_size).to(self.device)
        
        # copy params to target param
        for target_param, param in zip(self.target_value_net.parameters(), self.value_net.parameters()):
            target_param.data.copy_(param)
            
        # initialize optimizers 
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=self.critic_lr)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=self.critic_lr)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=v_lr)
        
        self.replay_buffer = ReplayBuffer(buffer_maxlen)
        
    def get_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        mean, log_std = self.actor.forward(state)
        std = log_std.exp()
        
        normal = Normal(mean, std)
        z = normal.sample()
        action = torch.tanh(z)
        action = action.cpu().detach().squeeze(0).numpy()
        
        return self.rescale_action(action)
    
    def rescale_action(self, action):
        return action * (self.action_range[1] - self.action_range[0]) / 2.0 +\
            (self.action_range[1] + self.action_range[0]) / 2.0
   
    def train_step(self, batch_size):
        # states, actions, rewards, next_states, _ = self.replay_buffer.sample(batch_size)
        states, actions, rewards, next_states, dones = self.replay_buffer.sample_batch(batch_size)
        states      = torch.FloatTensor(states).to(self.device)
        actions     = torch.FloatTensor(actions).to(self.device)
        rewards     = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones       = torch.FloatTensor(dones).to(self.device)
        dones = dones.view(dones.size(0), -1)
        
        curr_Q1s = self.critic1.forward(states, actions)
        curr_Q2s = self.critic2.forward(states, actions)
        next_V_targs = self.target_value_net(next_states)
        expected_Qs = rewards + (1 - dones) * self.gamma * next_V_targs
        
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
        
        next_actions, next_log_pi = self.actor.sample(next_states)
		
        next_Q1s = self.critic1(next_states, next_actions)
        next_Q2s = self.critic2(next_states, next_actions)
        curr_Vs = self.value_net.forward(states)
        
        # value Loss
        expected_Vs = torch.min(next_Q1s, next_Q2s) - next_log_pi
        v_loss = F.mse_loss(curr_Vs, expected_Vs.detach())
        
        # update value network and q networks
        self.value_optimizer.zero_grad()
        v_loss.backward()
        self.value_optimizer.step()
        
        # delayed update for policy net and target value nets
        new_actions, new_log_Pis = self.actor.sample(states)
        if self.update_step % self.delay_step == 0:
            expected_Ps = torch.min(
                self.critic1.forward(states, new_actions),
                self.critic2.forward(states, new_actions)
            )
            actor_loss = (new_log_Pis - expected_Ps).mean()
            
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            
            # target networks
            self.update_targets()

        self.update_step += 1
    
    def update_targets(self):
            for target_param, param in zip(self.target_value_net.parameters(), self.value_net.parameters()):
                target_param.data.copy_(self.tau * param + (1 - self.tau) * target_param)

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
        torch.save(self.critic1.state_dict(), "SAC2018_Q1_model.pth")
        torch.save(self.critic2.state_dict(), "SAC2018_Q2_model.pth")
        torch.save(self.value_net.state_dict(), "SAC2018_value_model.pth")
        torch.save(self.actor.state_dict(), "SAC2018_policy_model.pth")

if __name__ == "__main__":
    # 2018 agent
    # env = gym.make("Pendulum-v0")
    # actor_lr = 1e-3
    # critic_lr = 1e-3
    
    env_name = "Pendulum-v0"
    # set environment
    env = gym.make(env_name)
    env.seed(1)     # reproducible, general Policy gradient has high variance
    actor_lr = 3e-3
    critic_lr = 3e-3
    value_lr = 3e-3
    gamma = 0.99
    tau = 0.005
    delay_step = 2
    buffer_maxlen = 100000
    
    max_episodes = 200
    max_steps = 500
    batch_size = 32

    agent = SACAgent(env, gamma, tau, delay_step, value_lr, critic_lr, actor_lr, buffer_maxlen)
    
    agent.train()
    agent.save_model()
    
