import argparse
import datetime
import random
import gym
import numpy as np
import itertools
import torch
from collections import defaultdict

from pommerman import agents
from eda import trans_obs, compute_reward
from ppo.ppo import PPO, Transition
from ppo.replay_memory import ReplayMemory
from tensorboardX import SummaryWriter


parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
parser.add_argument('--env-name', default='OneVsOne-v0',
                    help='Pommerman Gym environment (default: PommeFFACompetition-v0)')
parser.add_argument('--policy', default="Deterministic",
                    help='Policy Type: Gaussian | Deterministic (default: Deterministic)')
parser.add_argument('--eval', type=bool, default=True,
                    help='Evaluates a policy a policy every 100 episode (default: True)')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor for reward (default: 0.99)')
parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                    help='target smoothing coefficient(τ) (default: 0.005)')
parser.add_argument('--lr', type=float, default=0.0003, metavar='G',
                    help='learning rate (default: 0.0003)')
parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
                    help='Temperature parameter α determines the relative importance of the entropy\
                            term against the reward (default: 0.2)')
parser.add_argument('--automatic_entropy_tuning', type=bool, default=True, metavar='G',
                    help='Automatically adjust α (default: False)')
parser.add_argument('--seed', type=int, default=123456, metavar='N',
                    help='random seed (default: 123456)')
parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                    help='batch size (default: 256)')
parser.add_argument('--num_steps', type=int, default=1000001, metavar='N',
                    help='maximum number of steps (default: 1000000)')
parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
                    help='hidden size (default: 256)')
parser.add_argument('--updates_per_step', type=int, default=1, metavar='N',
                    help='model updates per simulator step (default: 1)')
parser.add_argument('--start_steps', type=int, default=10000, metavar='N',
                    help='Steps sampling random actions (default: 10000)')
parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                    help='Value target update per no. of updates per step (default: 1)')
parser.add_argument('--replay_size', type=int, default=1000000, metavar='N',
                    help='size of replay buffer (default: 10000000)')
parser.add_argument('--cuda', action="store_true", default=False,
                    help='run on CUDA (default: False)')
args = parser.parse_args()

# Environment
env = gym.make(args.env_name)

# random seed
torch.manual_seed(args.seed)
np.random.seed(args.seed)
env.seed(args.seed)
random.seed(args.seed)

# Agent
ppo_agent = PPO(16, env.action_space.n)
# 1v1, two agents
agent_list = [ppo_agent, agents.SimpleAgent()]  # , agents.SimpleAgent(), agents.SimpleAgent()]
for idx, agent in enumerate(agent_list):
    assert isinstance(agent, agents.BaseAgent)
    agent.init_agent(idx, env.spec._kwargs['game_type'])

env.set_agents(agent_list)
env.set_init_game_state(None)
env.set_render_mode('human')
env.max_episode_steps = 800

print(env.action_space, env.observation_space)

# TesnorboardX
utc = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
writer = SummaryWriter(logdir=f'runs/{utc}_{ppo_agent.name}_{args.env_name}_{args.policy}')

# Memory
memory = ReplayMemory(args.replay_size)


# Training Loop
def train():
    total_step = 0
    for episode in itertools.count(1):
        episode_reward = 0
        episode_step = 0
        done = False
        state = env.reset()
        action_collect = defaultdict(int)
        while not done:
            # env.render()

            # this will call all agents' agent.act() method，
            actions = env.act(state)

            action_collect[actions[0]] += 1

            # env.step will execute all four agents' action(if alive), and return:
            #   next state: see in obs.py/obs_1v1.py
            #   rewards: type(rewards) == list, in FFA, len(rewards) == 4, and OneVsOne, len == 2
            #   done: whether this episode finished or not
            #   info: type(info) == dict, some extra info
            state_, rewards, done, _ = env.step(actions)

            # featurize state[0], feed into network
            obs, _ = trans_obs(state[0])
            # reward reshaping
            reward = compute_reward(state[0], state_[0], rewards, agent._character.bonus)
            agent._character.bonus = 0   # TODO this is a awful design, try to come up with sth. else.

            # not agent.is_alive(or reward[0] is -1), our agent dies, FFA, end this episode
            # or hit the time horizon.
            done = done or not env._agents[0].is_alive or episode_step + 1 == env.max_episode_steps

            # Ignore the "done" signal if it comes from hitting the time horizon.
            # that is, when it's an artificial terminal signal that isn't based on
            # the agent's state.
            # (https://github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py)
            mask = 1. if episode_step + 1 == env.max_episode_steps else float(not done)

            # Append transition to memory, for off-policy methods, can we use other agent's experience?
            # memory.push(obs, actions[0], ppo_agent.action_prob, reward, mask)

            trans = Transition(obs, actions[0], ppo_agent.action_prob, reward)
            ppo_agent.store_transition(trans)
            episode_step += 1
            total_step += 1
            episode_reward += reward
            if done and len(ppo_agent.buffer) > args.batch_size:
                ppo_agent.learn(memory, args.batch_size, writer=writer)

            state = state_

        writer.add_scalar('reward/train', episode_reward, episode)
        # print some info
        print(f"Episode: {episode}, total steps: {total_step}, episode steps: {episode_step},"
              f" reward: {round(episode_reward, 2)}, action: {dict(action_collect.items())}")
        # if total_step > args.num_steps:  # total train steps
        #     break
        if episode > 100000:
            break


def eval(train_episode):
    avg_reward = 0.
    eval_episodes = 10
    for _ in range(eval_episodes):
        state = env.reset()
        episode_reward = 0
        done = False
        while not done:
            actions = env.act(state)
            state_, reward, done, _ = env.step(actions)
            if not env._agents[0].is_alive:
                done = True
            episode_reward += reward[0]
            state = state_
        avg_reward += episode_reward
    avg_reward /= eval_episodes
    writer.add_scalar('avg_reward/test', avg_reward, train_episode)

    print("----------------------------------------")
    print("Test Episodes: {}, Avg. Reward: {}".format(eval_episodes, round(avg_reward, 2)))
    print("----------------------------------------")


train()
env.close()

