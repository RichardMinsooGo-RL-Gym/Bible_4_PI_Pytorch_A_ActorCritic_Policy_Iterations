import sys
IN_COLAB = "google.colab" in sys.modules

import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

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

# unlike A2CAgent in a2c.py, here I separated value and policy network.
class A2CAgent():
    
    def __init__(self, env, gamma, actor_lr, critic_lr):
        super(A2CAgent, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.env = env
        
        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.n
        
        self.gamma = gamma
        self.actor = Actor(self.state_size, self.action_size,
                           )
        self.critic = Critic(self.state_size)
        
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.critic_lr)
    
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
    
    def train_step(self, trajectory):
        critic_loss, actor_loss = self.compute_loss(trajectory)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
    
    def train(self):
        episode = 0
        while max_episodes > episode:
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
            self.train_step(trajectory)
            print("Episode " + str(episode+1) + ": " + str(episode_reward))
            episode += 1

    def save_model(self):
        torch.save(self.critic.state_dict(), "A2C_value_model.pth")
        torch.save(self.actor.state_dict(), "A2C_policy_model.pth")

if __name__ == "__main__":
    
    env_name = "CartPole-v0"
    # set environment
    env = gym.make(env_name)
    env.seed(1)     # reproducible, general Policy gradient has high variance
    actor_lr = 0.0005
    critic_lr = 0.001
    gamma = 0.99
    
    hidden_size = 256
    max_episodes = 500
    agent = A2CAgent(env, gamma, actor_lr, critic_lr)
    agent.train()
    agent.save_model()
    
