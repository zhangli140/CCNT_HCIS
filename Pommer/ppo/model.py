import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim=32):
        super(Actor, self).__init__()
        #   shape [8 x 8 x c]
        #              |
        #         [8 x 8 x 64]
        #              |
        #         [8 x 8 x 84]
        #          /        \
        #    [8 x 8 x 1]  [8 x 8 x 1]
        #         |            |
        #        [32]         [32]
        #         |            |
        #    [6(softmax)]  [1(tanh or None)]
        self.conv1 = nn.Conv2d(16, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 1, kernel_size=1)
        self.fc1 = nn.Linear(64, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_actions)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 64)
        x = F.relu(self.fc1(x))
        prob = F.softmax(self.fc2(x), dim=1)
        return prob  # (batch_size, num_actions)


class Critic(nn.Module):
    def __init__(self, num_inputs, hidden_dim):
        super(Critic, self).__init__()
        #   shape [8 x 8 x c]
        #              |
        #         [8 x 8 x 64]
        #              |
        #         [8 x 8 x 84]
        #          /        \
        #    [8 x 8 x 1]  [8 x 8 x 1]
        #         |            |
        #        [32]         [32]
        #         |            |
        #    [6(softmax)]  [1(tanh or None)]
        self.conv1 = nn.Conv2d(16, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 1, kernel_size=1)
        self.fc1 = nn.Linear(64, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 64)
        v = F.relu(self.fc1(x))
        v = self.fc2(v)
        return v


# TODO using a shared embedding net
class ActorCritic(nn.Module):
    def __init__(self, s_dim, a_dim):
        super(ActorCritic, self).__init__()
        #   shape [8 x 8 x c]
        #              |
        #         [8 x 8 x 64]
        #              |
        #         [8 x 8 x 84]
        #          /        \
        #    [8 x 8 x 1]  [8 x 8 x 1]
        #         |            |
        #        [32]         [32]
        #         |            |
        #    [6(softmax)]  [1(tanh or None)]
        self.conv1 = nn.Conv2d(16, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 1, kernel_size=1)
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, a_dim)

        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 1)

    def forward(self, x, info):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 64)
        pi = F.relu(self.fc1(x))
        prob = F.softmax(self.fc2(pi))

        v = F.relu(self.fc3(x))
        v = self.fc4(v)
        return prob, v
