from collections import namedtuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

import gym
import os
from pommerman.agents import BaseAgent
from eda import trans_obs
from ppo.model import Actor, Critic
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


Transition = namedtuple('Transition', ['state', 'action', 'a_log_prob', 'reward'])


class PPO(BaseAgent):
    name = 'PPO'

    def __init__(self, s_dim, a_dim,
                 gamma=0.99,
                 log_interval=10,
                 clip_param=0.2,
                 max_grad_norm=0.5,
                 ppo_update_time=10,
                 buffer_capacity=1000,
                 batch_size=32):
        super(PPO, self).__init__()
        self.gamma = gamma
        self.log_interval = log_interval
        self.clip_param = clip_param
        self.max_grad_norm = max_grad_norm
        self.ppo_update_time = ppo_update_time
        self.buffer_capacity = buffer_capacity  # TODO we want a buffer outside this class
        self.batch_size = batch_size

        self.actor_net = Actor(s_dim, a_dim, hidden_dim=256).to(device)
        self.critic_net = Critic(s_dim, hidden_dim=256).to(device)
        self.buffer = []
        self.counter = 0
        self.training_step = 0

        self.actor_optimizer = optim.Adam(self.actor_net.parameters(), lr=1e-3)
        self.critic_optimizer = optim.Adam(self.critic_net.parameters(), lr=3e-3)

    def select_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        with torch.no_grad():
            action_prob = self.actor_net(state)
        c = Categorical(action_prob)
        action = c.sample()
        return action.item(), action_prob[:, action.item()].item()

    def act(self, obs, action_space):
        obs, vec = trans_obs(obs)
        a, self.action_prob = self.select_action(obs)
        return a

    def sample_action(self, state):
        a, prob = self.select_action(state)
        return torch.tensor(a).long(), prob, a

    def get_value(self, state):
        state = torch.from_numpy(state).to(device)
        with torch.no_grad():
            value = self.critic_net(state)
        return value.item()

    # Save model parameters
    def save_model(self, env_name, suffix="", actor_path=None, critic_path=None):
        if not os.path.exists('models/'):
            os.makedirs('models/')

        if actor_path is None:
            actor_path = "models/sac_actor_{}_{}".format(env_name, suffix)
        if critic_path is None:
            critic_path = "models/sac_critic_{}_{}".format(env_name, suffix)
        print('Saving models to {} and {}'.format(actor_path, critic_path))
        torch.save(self.actor_net.state_dict(), actor_path)
        torch.save(self.critic_net.state_dict(), critic_path)

    # Load model parameters
    def load_model(self, actor_path, critic_path):
        print('Loading models from {} and {}'.format(actor_path, critic_path))
        if actor_path is not None:
            self.actor_net.load_state_dict(torch.load(actor_path))
        if critic_path is not None:
            self.critic_net.load_state_dict(torch.load(critic_path))

    def store_transition(self, transition):
        self.buffer.append(transition)
        self.counter += 1

    def learn(self, memo, batch_size, writer=None):
        state = torch.tensor([t.state for t in self.buffer], dtype=torch.float)
        action = torch.tensor([t.action for t in self.buffer], dtype=torch.long).view(-1, 1)
        reward = [t.reward for t in self.buffer]
        # update: don't need next_state
        # reward = torch.tensor([t.reward for t in self.buffer], dtype=torch.float).view(-1, 1)
        # next_state = torch.tensor([t.next_state for t in self.buffer], dtype=torch.float)
        old_action_prob = torch.tensor([t.a_log_prob for t in self.buffer], dtype=torch.float).view(-1, 1)

        R = 0
        Gt = []
        for r in reward[::-1]:
            R = r + self.gamma * R
            Gt.append(R)
        Gt.reverse()
        Gt = torch.tensor(Gt, dtype=torch.float)

        for i in range(self.ppo_update_time):
            for index in BatchSampler(SubsetRandomSampler(range(len(self.buffer))), self.batch_size, drop_last=False):
                # 每次sample一个batch, 一次学习恰好sample完所有的transition
                if self.training_step % 10000 == 0:
                    self.save_model('PommOneVsOne', suffix=str(self.training_step))

                # with torch.no_grad():
                Gt_index = Gt[index].view(-1, 1)
                V = self.critic_net(state[index].to(device))
                delta = Gt_index - V
                advantage = delta.detach()
                # epoch iteration, PPO core!!!
                action_prob = self.actor_net(state[index].to(device)).gather(1, action[index])  # new policy

                ratio = torch.exp(torch.log(action_prob) - torch.log(old_action_prob[index]))
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * advantage

                # update actor network
                action_loss = -torch.min(surr1, surr2).mean()

                writer.add_scalar('loss/action_loss', action_loss.item(), global_step=self.training_step)
                self.actor_optimizer.zero_grad()
                action_loss.backward()
                nn.utils.clip_grad_norm_(self.actor_net.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()

                # update critic network
                value_loss = F.mse_loss(Gt_index, V)
                writer.add_scalar('loss/value_loss', value_loss.item(), global_step=self.training_step)
                self.critic_optimizer.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(self.critic_net.parameters(), self.max_grad_norm)
                self.critic_optimizer.step()
                self.training_step += 1

        del self.buffer[:]  # clear experience


def run_cart_pole(render=False):
    seed = 1  # TODO set_seed func would be better
    env = gym.make('CartPole-v0').unwrapped
    torch.manual_seed(seed)
    env.seed(seed)

    agent = PPO(4, 2)
    for episode in range(1000):
        state, done = env.reset(), False
        epi_step = 0
        while not done:
            action, action_prob = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            trans = Transition(state, action, action_prob, reward, next_state)
            if render:
                env.render()
            agent.store_transition(trans)
            state = next_state
            epi_step += 1
            if done:
                if len(agent.buffer) >= agent.batch_size:
                    agent.learn(None, None, updates=episode)
                print(f'Epi:{episode}, reward:{epi_step}')
                # agent.writer.add_scalar('liveTime/livestep', epi_step, global_step=episode)


if __name__ == '__main__':
    run_cart_pole()
    print("end")