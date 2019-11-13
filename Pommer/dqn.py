"""
This part of code is the DQN brain, which is a brain of the agent.
All decisions are made in here.
Using Tensorflow to build the neural network.
View more on my tutorial page: https://morvanzhou.github.io/tutorials/
Using:
Tensorflow: 1.0
gym: 0.8.0
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sacd.utils import hard_update
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.normal_(m.weight, mean=0, std=0.3)
        torch.nn.init.constant_(m.bias, 0.1)


class QNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim):
        super(QNetwork, self).__init__()

        # Q1 architecture
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, num_actions)

        self.apply(weights_init_)

    def forward(self, state):
        # print(state.shape) [256, 366]
        xu = state

        x = F.relu(self.linear1(xu))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)

        return x

# Deep Q Network off-policy
class DQN:
    def __init__(
            self,
            n_actions,
            n_features,
            learning_rate=0.01,
            reward_decay=0.9,
            e_greedy=0.9,
            replace_target_iter=300,
            memory_size=500,
            batch_size=32,
            e_greedy_increment=None,
            output_graph=False,
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max

        # total learning step
        self.learn_step_counter = 0

        self.cost_his = []
        self.eval = QNetwork(n_features, n_actions, hidden_dim=20)
        self.next = QNetwork(n_features, n_actions, hidden_dim=20)
        hard_update(self.next, self.eval)
        self.next.eval()
        self.optimizer = optim.Adam(self.eval.parameters(), self.lr)
        self.criterion = nn.MSELoss()

    def sample_action(self, obs):
        # to have batch dimension when feed into tf placeholder
        obs = obs[np.newaxis, :]
        obs = torch.FloatTensor(obs)
        if np.random.uniform() < self.epsilon:
            # forward feed the observation and get q value for every actions
            actions_value = self.eval(obs)
            action = torch.argmax(actions_value, dim=1)
        else:
            action = torch.tensor(np.random.randint(0, self.n_actions))
        return action, None, action

    def learn(self, memo, batch_size=32, updates=None):
        if self.learn_step_counter % 300 == 0:
            hard_update(self.next, self.eval)
            print('\ntarget_net state dict replaced!')

        s, action, reward, s_, done = memo.sample(batch_size=batch_size)
        s = torch.FloatTensor(s).to(device)
        s_ = torch.FloatTensor(s_).to(device)
        action = torch.FloatTensor(action).to(device)
        reward = torch.FloatTensor(reward).to(device)
        done = torch.FloatTensor(done).to(device).unsqueeze(1)
        # get q_target
        q_eval = self.eval(s)
        q_next = self.next(s_)

        with torch.no_grad():
            q_target = reward + self.gamma * torch.max(q_next, dim=1)[0] * (1 - done)

        q_eval = torch.gather(q_eval, dim=1, index=action.view(-1, 1).long())

        # shape (batch_size, ), (batch_size, )
        loss = self.criterion(q_eval.view(-1), q_target)

        self.optimizer.zero_grad()
        loss.backward()
        # for param in self.eval.parameters():
        #     param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        # sum_writer.add_scalar('scalar/loss', loss.item(), self.train_step)
        self.cost_his.append(loss.cpu().item())

        # increasing epsilon
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1
        return None, None, None, None, loss.cpu().item()

    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()