import os
import sys
import time
import math

import gym
import torch
import numpy as np
import torch.nn.functional as F


import colorama
from pommerman import agents
from collections import Counter
from SAC.SACDiscrete import SACDiscrete
from SAC.Config import config

ROLLOUTS_PER_BATCH = 1
batch = []
RENDER = True


def make_env(agent_list):
    env = gym.make('PommeFFACompetition-v0')

    for idx, agent in enumerate(agent_list):
        assert isinstance(agent, agents.BaseAgent)
        agent.init_agent(idx, env.spec._kwargs['game_type'])

    env.set_agents(agent_list)
    env.set_init_game_state(None)
    env.set_render_mode('human')
    return env


class World:
    def __init__(self, init_gmodel=True):
        config.environment = None
        self.agent = SACDiscrete(s_dim=(11,11,14), a_dim=6, config=config)
        # self.stoner = Stoner()

        self.agent_list = [
            self.agent,
            # self.stoner
            agents.SimpleAgent(),
            agents.SimpleAgent(),
            agents.SimpleAgent()
        ]
        self.env = make_env(self.agent_list)  # FFA
        self.agent.setup_env(self.env, name='PommeFFA')
        fmt = {
            'int': self.color_sign,
            'float': self.color_sign
        }
        np.set_printoptions(formatter=fmt, linewidth=300)

    def color_sign(self, x):
        if x == 0:
            c = colorama.Fore.LIGHTBLACK_EX
        elif x == 1:
            c = colorama.Fore.BLACK
        elif x == 2:
            c = colorama.Fore.BLUE
        elif x == 3:
            c = colorama.Fore.RED
        elif x == 4:
            c = colorama.Fore.RED
        elif x == 10:
            c = colorama.Fore.YELLOW
        else:
            c = colorama.Fore.WHITE
        x = '{0: <2}'.format(x)
        return f'{c}{x}{colorama.Fore.RESET}'


def do_rollout(env, agent, do_print=False):
    done, state = False, env.reset()
    rewards, dones = [], []
    states, actions, hidden, probs, values = agent.clear()

    while not done:
        if do_print:
            time.sleep(0.1)
            os.system('clear')
            print(state[0]['board'])
        if RENDER:
            env.render()
        action = env.act(state)
        state, reward, done, info = env.step(action)
        if not env._agents[0].is_alive:
            done = True
        # if reward[0] == -1: done = True
        rewards.append(reward[0])
        dones.append(done)

    hidden = hidden[:-1].copy()
    hns, cns = [], []
    for hns_cns_tuple in hidden:
        hns.append(hns_cns_tuple[0])
        cns.append(hns_cns_tuple[1])

    return states.copy(), actions.copy(), rewards, dones, (hns, cns), probs.copy(), values.copy()


def train(world):

    sac_agent, env = world.agent, world.env
    sac_agent = SACDiscrete(s_dim=(11,11,14), a_dim=6, config=config)  # TODO 删掉
    # load model

    if os.path.exists("training.txt"):
        os.remove("training.txt")
    max_episodes = 40000
    rr = -1
    ii = 0
    for i in range(max_episodes):
        done, obs = False, env.reset()
        rewards, dones = [], []
        states, actions, hidden, probs, values = sac_agent.clear()

        # sac_agent.step()
        full_rollouts = [do_rollout(env, sac_agent) for _ in range(ROLLOUTS_PER_BATCH)]


def run(world):
    done, ded, state, _ = False, False, world.env.reset(), world.leif.clear()

    while not done:
        action = world.env.act(state)
        state, reward, done, info = world.env.step(action)
        print(world.leif.board_cent)
        print(world.leif.bbs_cent)
        print(world.leif.bl_cent)
        time.sleep(0.2)

    world.env.close()
    return None


def eval(world, init_gmodel=False):
    env = world.env
    model = world.model
    leif = world.leif
    leif.debug = True
    leif.stochastic = False

    do_print = False
    done = None
    reward = 0
    last_reward = [0, 0, 0, 0]

    while True:
        model.load_state_dict(torch.load("convrnn-s.weights", map_location='cpu'))

        done, state, _ = False, env.reset(), leif.clear()
        t = 0
        while not done:
            if do_print:
                time.sleep(0.1)
                os.system('clear')
                print(state[0]['board'])
                print("\n\n")
                print("Probs: \t", leif.probs[-1] if len(leif.probs) > 0 else [])
                print("Val: \t", leif.values[-1] if len(leif.values) > 0 else None)
                print("\nLast reward: ", last_reward, "Time", t)
            env.render()
            action = env.act(state)
            print(action)
            time.sleep(0.5)
            state, reward, done, info = env.step(action)
            if reward[0] == -1:
                last_reward = reward
                break
            t += 1


def readme(world):
    print("Usage: ")
    print("\t to train:\tpython main.py train")
    print("\t to evaluate:\tpython main.py eval\n\n")
    print("Procedure:")
    print("Start the training. Wait for 300 episodes (this will generate weights file)."
          " Run evaluate. See running results.")


entrypoint = next(iter(sys.argv[1:]), "readme")
locals()[entrypoint](World())
