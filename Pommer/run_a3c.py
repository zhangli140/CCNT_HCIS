import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.distributions import Categorical

import pommerman
from pommerman import agents
from A3C.sharedAdam import SharedAdam

import os
# on windows, multiprocessing: https://pytorch.org/docs/stable/notes/windows.html

# use one thread for parallel as they will block
# each other otherwise (https://github.com/ikostrikov/pytorch-a3c/issues/33)
os.environ["OMP_NUM_THREADS"] = "1"
mp = mp.get_context('spawn')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class StaticAgent(agents.BaseAgent):
    """ Static agent"""

    def act(self, obs, action_space):
        return pommerman.constants.Action.Stop.value

# TODO I hate using global constant variables
UPDATE_GLOBAL_ITER = 800
GAMMA = 0.95
LAMBDA = 1
MAX_EP = 60000
LEARNING_RATE = 0.00001
eps = np.finfo(np.float32).eps.item()


def load_checkpoint(filename, model, optimizer):
    if os.path.isfile(filename):
        print("# loading checkpoint '{}'".format(filename))
        checkpoint = torch.load(filename)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("# loaded checkpoint '{}'".format(filename))
    else:
        print("# no checkpoint found at '{}'".format(filename))
    return model, optimizer


def save_checkpoint(filename, model, optimizer):
    state = {
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    torch.save(state, filename)


def log(global_ep, global_ep_r, ep_r, res_queue, global_nr_steps, nr_steps, name):
    with global_ep.get_lock():
        global_ep.value += 1
    with global_ep_r.get_lock():
        if global_ep_r.value == 0.:
            global_ep_r.value = -1  # ep_r
            global_nr_steps.value = nr_steps
        else:
            global_ep_r.value = global_ep_r.value * 0.99 + ep_r * 0.01
            global_nr_steps.value = global_nr_steps.value * 0.99 + nr_steps * 0.01
    res_queue.put(ep_r)
    print(
        name,
        "Ep:", global_ep.value,
        "| Avg Ep_r: %.2f" % global_ep_r.value,
        "| Avg Steps: %d" % global_nr_steps.value,
        "| Ep_r / Steps: %.2f" % (ep_r / nr_steps),
        # "| Ep_r: %.2f" % ep_r,
    )


def update_glob_net(opt, lnet, gnet, agent, GAMMA):

    def ensure_shared_grads(lnet, gnet):
        for param, shared_param in zip(lnet.parameters(), gnet.parameters()):
            if shared_param.grad is not None:
                return
            shared_param._grad = param.grad

    # print(list(map(len, [agent.rewards, agent.values, agent.logProbs, agent.tps])))
    # every value in agent.values is tensor with shape=(1,1)
    # every log_prob in agent.logProbs is tensor with shape=(1, 1), 这里的 shape 不一致
    # every entropy in agent.entropies is tensor with shape=(1,)
    # every tp in agent.tps is tensor with shape=(1, 1)

    R = 0
    actor_loss = 0
    value_loss = 0
    # about GAE: Generalized Advantage Estimator (https://zhuanlan.zhihu.com/p/45107835)
    gae = 0
    agent.values.append(torch.zeros(1))  # we need to add this for the deltaT equation(below)
    # print(agent.rewards)
    for i in reversed(range(len(agent.rewards))):
        R = GAMMA * R + agent.rewards[i]
        advantage = R - agent.values[i]
        value_loss = value_loss + 0.5 * advantage.pow(2)
        deltaT = agent.rewards[i] + GAMMA * agent.values[i + 1].data - agent.values[i].data
        gae = gae * GAMMA * LAMBDA + deltaT  # generalized advantage estimator
        #
        actor_loss = actor_loss - agent.logProbs[i] * gae - 0.01 * agent.entropies[i]

    # TODO: tp_loss
    tp_loss = 0.
    for i in range(len(agent.tps)):
        tp_loss += (agent.tps[i] - i/len(agent.tps)) ** 2

    tp_loss /= len(agent.tps)

    loss = (actor_loss + 0.5 * value_loss + 0.5 * tp_loss)
    opt.zero_grad()
    loss.backward(retain_graph=True)
    ensure_shared_grads(lnet, gnet)
    opt.step()
    lnet.load_state_dict(gnet.state_dict())
    lnet.zero_grad()
    agent.clear_actions()


class A3CNet(nn.Module):
    def __init__(self, s_dim=16, a_dim=6):
        super(A3CNet, self).__init__()
        #   shape [8 x 8 x c], in this case c = 16
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
        self.fc1 = nn.Linear(64, 128)

        self.a_head = nn.Linear(128, a_dim)
        self.v_head = nn.Linear(128, 1)
        self.tp_head = nn.Linear(128, 1)

    def forward(self, x, info=None):
        # x.shape = (-1, 16, 8, 8)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 64)
        x = F.relu(self.fc1(x))
        prob = F.softmax(self.a_head(x), dim=1)

        v = self.v_head(x)
        tp = self.tp_head(x)
        return prob, v, tp


class A3CAgent(agents.BaseAgent):
    def __init__(self, model):
        super(A3CAgent, self).__init__()
        self.model = model
        self.rewards = []
        self.tps = []
        self.values = []
        self.logProbs = []
        self.entropies = []
        self.train = True

    def act(self, state, action_space):
        # TODO vec was not used
        obs, vec = self.feature(state)
        obs = torch.from_numpy(obs).unsqueeze(0)
        prob, value, tp = self.model(obs)
        log_prob = F.log_softmax(prob, dim=1)
        entropy = -(log_prob * prob).sum(1)

        c = Categorical(prob)
        action = c.sample()

        log_prob = log_prob.gather(dim=1, index=action.unsqueeze(0))
        a = action.item()

        if self.train:
            self.tps.append(tp)
            self.entropies.append(entropy)
            self.values.append(value)
            self.logProbs.append(log_prob)

        return a

    def add_reward(self, reward):
        clipped_reward = max(min(reward, 1), -1)
        self.rewards.append(clipped_reward)

    @staticmethod
    def feature(obs):
        o = np.zeros(shape=(16, 8, 8), dtype=np.float32)
        board = obs['board']
        for i in range(12):
            o[i][board == i] = 1.
        o[12] = obs['bomb_blast_strength']
        o[13] = obs['bomb_life']
        o[14] = obs['bomb_moving_direction']
        o[15] = obs['flame_life']

        vec = np.array([obs['ammo'], obs['can_kick'], obs['blast_strength'], obs['step_count'] / 800], dtype=np.float32)
        return o, vec

    def clear_actions(self):
        self.values = []
        self.logProbs = []
        self.rewards = []
        self.entropies = []
        self.tps = []


class Worker(mp.Process):
    def __init__(self, gnet, optimizer, global_ep, global_ep_r, global_nr_steps, res_queue, name: int):
        super(Worker, self).__init__()
        self.agent_idx = 0
        self.name = 'w%s' % name
        print(f'Worker {name} init called.')
        self.g_ep, self.g_ep_r, self.res_queue = global_ep, global_ep_r, res_queue
        self.gnet, self.opt = gnet, optimizer
        self.lnet = A3CNet()
        self.lnet.load_state_dict(gnet.state_dict())
        # self.lnet.train()
        self.lnet.zero_grad()
        self.results = []
        self.global_nr_steps = global_nr_steps
        self.empty_oh_action = np.zeros((6, 1))
        self.saved_oh_actions = np.repeat(self.empty_oh_action, 6, 1)

        self.A3CAgent = None
        self.env = None
        self.agentList = None

    def run(self):
        # If we move this to "init", we get an error on recursion depth
        print(f'Worker {self.name} starts to run.')
        self.A3CAgent = A3CAgent(self.lnet)
        self.agentList = [self.A3CAgent, StaticAgent()]
        self.env = env = pommerman.make('OneVsOne-v0', self.agentList)
        print(self.env)
        total_step = 1
        while self.g_ep.value < MAX_EP:
            # Step 2). worker interacts with environment
            s_act = self.env.reset()
            max_ammo = old_max_ammo = 1
            ep_r = 0.
            render = False  # self.g_ep.value % 20==0
            if self.name == 'w0':
                # TODO: use tensorboard to show some params of worker-0
                pass
            while True:
                # only render worker 0
                if self.name == 'w0' and render:
                    self.env.render()
                agent_actions = self.env.act(s_act)
                a = agent_actions[self.agent_idx]
                self.saved_oh_actions[:, :-1] = self.saved_oh_actions[:, 1:]  # time shift
                self.saved_oh_actions[:, -1] = self.empty_oh_action[:, 0]  # erase last value
                self.saved_oh_actions[a, -1] = 1  # insert new one-hot

                s_new, rewards, done, _ = self.env.step(agent_actions)
                # not(10 in s_new[self.agent_nr]['alive']) #if done or agent 10 is dead
                done = done or rewards[self.agent_idx] == -1
                max_ammo = max(max_ammo, s_act[self.agent_idx]['ammo'])
                # reward and buffer
                r = rewards[self.agent_idx]
                # if (10 in s_act[self.agent_nr]['alive']) and total_step!=1:
                #    r = get_reward(s_new,s_act,self.agent_nr,r,max_ammo,old_max_ammo,a,a_old,self.saved_oh_actions)
                ep_r += r
                self.A3CAgent.add_reward(r)
                if total_step % UPDATE_GLOBAL_ITER == 0 or done:  # update global and assign to local net
                    update_glob_net(self.opt, self.lnet, self.gnet, self.A3CAgent, GAMMA)
                    if done:
                        log(self.g_ep, self.g_ep_r, ep_r, self.res_queue, self.global_nr_steps,
                            s_new[self.agent_idx]['step_count'], self.name)
                        break
                s_act = s_new
                old_max_ammo = max_ammo
                a_old = a
                total_step += 1
        self.res_queue.put(None)


def main():
    print('Initialize global network')
    global_net = A3CNet()  # global network
    # global_net.train()  # Set in training mode, only affect BN, Dropout etc.

    filename = './A3C_TP_trained_critic.pth'

    global_net.share_memory()  # share the global parameters in multiprocessing
    optimizer = SharedAdam(global_net.parameters(), lr=LEARNING_RATE)  # global optimizer
    load_checkpoint(filename, global_net, optimizer)

    for g in optimizer.param_groups:
        g['lr'] = LEARNING_RATE   # setting up lr

    global_ep, global_ep_r, global_nr_steps, res_queue = \
        mp.Value('d', 0), mp.Value('d', 0.), mp.Value('d', 0.), mp.Queue()

    # parallel training
    print('Initialize workers')
    workers = [Worker(global_net, optimizer, global_ep, global_ep_r, global_nr_steps, res_queue, i)  # for i in range(1)]
               for i in range(mp.cpu_count())]

    print('Start workers')
    [w.start() for w in workers]
    res = []  # record episode reward to plot
    while True:
        r = res_queue.get()
        if r is None:
            break
        res.append(r)

    filename = './A3C_TP_trained_critic_actor_1.pth'
    save_checkpoint(filename, global_net, optimizer)

    with open('./A3C_TP_trained_critic_actor.txt', 'a') as f:
        for item in res:
            f.write("%s\n" % item)

    print('joining workers')
    [w.join() for w in workers]


if __name__ == '__main__':
    main()
