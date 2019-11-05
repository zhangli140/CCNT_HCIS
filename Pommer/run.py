import argparse
import datetime
import gym
import numpy as np
import itertools
import torch
from collections import defaultdict

from pommerman import agents
from sacd.sac import SAC
from tensorboardX import SummaryWriter
from sacd.replay_memory import ReplayMemory

# TODO
#   loss compute [discrete sac]
#   reward reshaping
#   for agents, act was a callback function.
#   train/test need to be separated

parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
parser.add_argument('--env-name', default='PommeFFACompetition-v0',
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
parser.add_argument('--cuda', action="store_true",
                    help='run on CUDA (default: False)')
args = parser.parse_args()

# Environment
env = gym.make(args.env_name)
torch.manual_seed(args.seed)
np.random.seed(args.seed)
env.seed(args.seed)
print(env.action_space, env.observation_space)

# Agent
# TODO
sac_agent = SAC(366, env.action_space, args)

agent_list = [sac_agent, agents.SimpleAgent(), agents.SimpleAgent(), agents.SimpleAgent()]
for idx, agent in enumerate(agent_list):
    assert isinstance(agent, agents.BaseAgent)
    agent.init_agent(idx, env.spec._kwargs['game_type'])

env.set_agents(agent_list)
env.set_init_game_state(None)
env.set_render_mode('human')
env.max_episode_steps = 800
# TesnorboardX
writer = SummaryWriter(logdir='runs/{}_SAC_{}_{}_{}'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
                                                            args.env_name,
                                                            args.policy,
                                                            "autotune" if args.automatic_entropy_tuning else ""))

# Memory
memory = ReplayMemory(args.replay_size)


# Training Loop
def train():
    total_step = 0
    updates = 0
    for episode in itertools.count(1):
        episode_reward = 0
        episode_step = 0
        done = False
        state = env.reset()
        action_collect = defaultdict(int)
        while not done:
            if len(memory) > args.batch_size:
                # TODO should do this in a member function of sac_agent
                # Number of updates per step in environment
                for i in range(args.updates_per_step):
                    # Update parameters of all the networks
                    critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha \
                        = sac_agent.update_parameters(memory, args.batch_size, updates)

                    writer.add_scalar('loss/critic_1', critic_1_loss, updates)
                    writer.add_scalar('loss/critic_2', critic_2_loss, updates)
                    writer.add_scalar('loss/policy', policy_loss, updates)
                    writer.add_scalar('loss/entropy_loss', ent_loss, updates)
                    writer.add_scalar('entropy_temprature/alpha', alpha, updates)
                    updates += 1
                    if updates % 10000 == 0:
                        sac_agent.save_model('PommFFA', suffix=str(episode)+'_'+str(updates))

            obs = sac_agent._translate_obs(state[0])
            actions = env.act(state)  # this will call all four agents' agent.act()
            action_collect[actions[0]] += 1
            state_, rewards, done, _ = env.step(actions)  # this execute all four agents' action(if alive)
            # type(rewards) == list, in FFA, len(rewards) == 4, and OneVsOne, len == 2

            # not sac_agent.is_alive(or reward[0] is -1), our agent dies, FFA, end the episode
            if not env._agents[0].is_alive:
                done = True
            episode_step += 1
            total_step += 1
            episode_reward += rewards[0]
            obs_ = sac_agent._translate_obs(state_[0])

            # Ignore the "done" signal if it comes from hitting the time horizon.
            # (https://github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py)
            mask = 1. if episode_step == env.max_episode_steps else float(not done)

            # Append transition to memory, since sac is off-policy, can we use other agent's experience?
            memory.push(obs, actions[0], rewards[0], obs_, mask)
            state = state_

        if total_step > args.num_steps:  # total train steps
            break
        # info print
        writer.add_scalar('reward/train', episode_reward, episode)
        print(f"Episode: {episode}, total steps: {total_step}, "
              f"episode steps: {episode_step}, reward: {round(episode_reward, 2)},"
              f" action: {dict(action_collect.items())}")


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

# total_numsteps = 0
# updates = 0
#
# for i_episode in itertools.count(1):
#     episode_reward = 0
#     episode_steps = 0
#     done = False
#     state = env.reset()
#
#     while not done:
#         if args.start_steps > total_numsteps:
#             action = env.action_space.sample()  # Sample random action
#         else:
#             action = sac_agent.select_action(state)  # Sample action from policy
#
#         if len(memory) > args.batch_size:
#             # Number of updates per step in environment
#             for i in range(args.updates_per_step):
#                 # Update parameters of all the networks
#                 critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha \
#                     = sac_agent.update_parameters(memory, args.batch_size, updates)
#
#                 writer.add_scalar('loss/critic_1', critic_1_loss, updates)
#                 writer.add_scalar('loss/critic_2', critic_2_loss, updates)
#                 writer.add_scalar('loss/policy', policy_loss, updates)
#                 writer.add_scalar('loss/entropy_loss', ent_loss, updates)
#                 writer.add_scalar('entropy_temprature/alpha', alpha, updates)
#                 updates += 1
#
#         obs = sac_agent._translate_obs(state[0])
#         actions = env.act(state)
#         # actions[0] = action
#         next_state, reward, done, _ = env.step(actions)  # Step
#         if not env._agents[0].is_alive:  # early stop, FFA, When our agent dies, end game
#             done = True
#         episode_steps += 1
#         total_numsteps += 1
#         episode_reward += reward[0]
#         obs_ = sac_agent._translate_obs(next_state[0])
#         # Ignore the "done" signal if it comes from hitting the time horizon.
#         # (https://github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py)
#         mask = 1. if episode_steps == env.max_episode_steps else float(not done)
#
#         memory.push(obs, action, reward[0], obs_, mask)  # Append transition to memory
#
#         state = next_state
#
#     if total_numsteps > args.num_steps:
#         break
#
#     writer.add_scalar('reward/train', episode_reward, i_episode)
#     print("Episode: {}, total numsteps: {}, episode steps: {}, reward: {}".format(i_episode, total_numsteps,
#                                                                                   episode_steps,
#                                                                                   round(episode_reward, 2)))
#
#     if i_episode % 100 == 0 and args.eval:
#         avg_reward = 0.
#         episodes = 10
#         for _ in range(episodes):
#             state = env.reset()
#             episode_reward = 0
#             done = False
#             while not done:
#                 # action = sac_agent.act(state[0], action_space=None, eval=True)
#                 actions = env.act(state)
#                 next_state, reward, done, _ = env.step(actions)
#                 if reward[0] == -1:
#                     done = True
#                 episode_reward += reward[0]
#
#                 state = next_state
#             avg_reward += episode_reward
#         avg_reward /= episodes
#
#         writer.add_scalar('avg_reward/test', avg_reward, i_episode)
#
#         print("----------------------------------------")
#         print("Test Episodes: {}, Avg. Reward: {}".format(episodes, round(avg_reward, 2)))
#         print("----------------------------------------")
#
# env.close()
