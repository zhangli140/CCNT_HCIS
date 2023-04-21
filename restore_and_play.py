from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from abc import ABC

from absl import app
from absl import flags
from absl import logging
import tensorflow as tf

from open_spiel.python import policy
from open_spiel.python import rl_environment
from open_spiel.python.algorithms import exploitability
# from open_spiel.python.algorithms import nfsp
from open_spiel.python.algorithms import random_agent
import arm_tf
import numpy as np
import matplotlib.pyplot as plt
import datetime
import time
import nfsp_arm
import pickle
import nfsp

FLAGS = flags.FLAGS

flags.DEFINE_integer("num_train_episodes", int(20000),  # 15e6
                     "Number of training episodes.")
flags.DEFINE_integer("eval_every", 10000,
                     "Episode frequency at which the agents are evaluated.")
# flags.DEFINE_list("hidden_layers_sizes", [
#     128, 64
# ], "Number of hidden units in the avg-net and Q-net.")
flags.DEFINE_list("hidden_layers_sizes", [
    64, 128, 64
], "Number of hidden units in the avg-net and Q-net.")
flags.DEFINE_integer("replay_buffer_capacity", int(3000),
                     "Size of the replay buffer.")
flags.DEFINE_integer("reservoir_buffer_capacity", int(2e6),
                     "Size of the reservoir buffer.")
flags.DEFINE_float("anticipatory_param", 0.5,  # 0.1 origin
                   "Prob of using the rl best response as episode policy.")


class NFSPPolicies(policy.Policy, ABC):
    """Joint policy to be evaluated."""

    def __init__(self, env, nfsp_policies, mode):
        game = env.game
        player_ids = [0, 1]
        super(NFSPPolicies, self).__init__(game, player_ids)
        self._policies = nfsp_policies
        self._mode = mode
        self._obs = {"info_state": [None, None], "legal_actions": [None, None]}

    def action_probabilities(self, state, player_id=None):
        cur_player = state.current_player()
        legal_actions = state.legal_actions(cur_player)

        self._obs["current_player"] = cur_player
        self._obs["info_state"][cur_player] = (
            state.information_state_tensor(cur_player))
        self._obs["legal_actions"][cur_player] = legal_actions

        info_state = rl_environment.TimeStep(
            observations=self._obs, rewards=None, discounts=None, step_type=None)

        # if cur_player == 0:
        with self._policies[cur_player].temp_mode_as(self._mode):
            # print(self._policies[cur_player].mode)
            p = self._policies[cur_player].step(info_state, is_evaluation=True).probs
        # else:
        #     p = self._policies[cur_player].step(info_state, is_evaluation=True).probs
        prob_dict = {action: p[action] for action in legal_actions}
        # print(prob_dict)
        return prob_dict


class ARMPolicies(policy.Policy, ABC):
    """Joint policy to be evaluated."""

    def __init__(self, env, arm_policies, mode):
        game = env.game
        player_ids = [0, 1]
        super(ARMPolicies, self).__init__(game, player_ids)
        self._policies = arm_policies
        self._mode = mode
        self._obs = {"info_state": [None, None], "legal_actions": [None, None]}

    def action_probabilities(self, state, player_id=None):
        cur_player = state.current_player()
        legal_actions = state.legal_actions(cur_player)

        self._obs["current_player"] = cur_player
        self._obs["info_state"][cur_player] = (
            state.information_state_tensor(cur_player))
        self._obs["legal_actions"][cur_player] = legal_actions

        info_state = rl_environment.TimeStep(
            observations=self._obs, rewards=None, discounts=None, step_type=None)

        p = self._policies[cur_player].step(info_state, is_evaluation=True).probs
        prob_dict = {action: p[action] for action in legal_actions}
        return prob_dict


def eval_against_random_bots(env, trained_agents, random_agents, num_episodes):
    """Evaluates `trained_agents` against `random_agents` for `num_episodes`."""
    num_players = len(trained_agents)
    sum_episode_rewards = np.zeros(num_players)
    win_rate = np.zeros(num_players)
    for player_pos in range(num_players):
        count_vic = 0
        cur_agents = random_agents[:]
        cur_agents[player_pos] = trained_agents[player_pos]
        for _ in range(num_episodes):
            time_step = env.reset()
            episode_rewards = 0
            while not time_step.last():
                player_id = time_step.observations["current_player"]
                if player_pos == player_id:
                    with cur_agents[player_id].temp_mode_as(nfsp_arm.MODE.average_policy):
                        agent_output = cur_agents[player_id].step(time_step, is_evaluation=True)
                else:
                    agent_output = cur_agents[player_id].step(time_step, is_evaluation=True)
                action_list = [agent_output.action]
                time_step = env.step(action_list)
                episode_rewards += time_step.rewards[player_pos]
            if episode_rewards > 0:
                episode_rewards = 1.0
                count_vic = count_vic + 1
            elif episode_rewards < 0:
                episode_rewards = -1.0
            elif episode_rewards == 0:
                episode_rewards = 0
                count_vic = count_vic + 1
            sum_episode_rewards[player_pos] += episode_rewards
            win_rate[player_pos] = count_vic
    return sum_episode_rewards / num_episodes, win_rate / num_episodes


def eval_against_random_bots_nfsp_arm(env, trained_agents, random_agents, num_episodes):
    """Evaluates `trained_agents` against `random_agents` for `num_episodes`."""
    num_players = len(trained_agents)
    sum_episode_rewards = np.zeros(num_players)
    for player_pos in range(num_players):
        cur_agents = random_agents[:]
        cur_agents[player_pos] = trained_agents[player_pos]
        for _ in range(num_episodes):
            time_step = env.reset()
            episode_rewards = 0
            while not time_step.last():
                player_id = time_step.observations["current_player"]
                if hasattr(cur_agents[player_id], 'tag') and cur_agents[player_id].tag() == "nfsp_arm":
                    with cur_agents[player_id].temp_mode_as(nfsp_arm.MODE.average_policy):
                        agent_output = cur_agents[player_id].step(time_step, is_evaluation=True)
                else:
                    # print(cur_agents[player_id])
                    with cur_agents[player_id].temp_mode_as(nfsp.MODE.average_policy):
                        agent_output = cur_agents[player_id].step(time_step, is_evaluation=True)
                action_list = [agent_output.action]
                time_step = env.step(action_list)
                episode_rewards += time_step.rewards[player_pos]
            # log every episode_rewards
            if hasattr(cur_agents[0], 'tag') and cur_agents[0].tag() == "nfsp_arm":
                with open('log/main_nfsp_arm_0_vs_nfsp1_every.txt', 'a+') as f:
                    f.write('{}\n'.format(episode_rewards))
            else:
                with open('log/main_nfsp_arm_1_vs_nfsp0_every.txt', 'a+') as f:
                    f.write('{}\n'.format(episode_rewards))
            sum_episode_rewards[player_pos] += episode_rewards
    return sum_episode_rewards / num_episodes


class ImportNFSP:
    """  Importing and running isolated TF graph """

    def __init__(self, info_state_size, num_actions, hidden_layers_sizes, loc):
        # Create local graph and use it in the session
        # origin nfsp
        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph)
        reservoir_buffer_capacity = 2e6
        anticipatory_param = 0.5
        num_players = 2
        kwargs = {
            "replay_buffer_capacity": 200000,
            "epsilon_decay_duration": 30000,
            "epsilon_start": 0.06,
            "epsilon_end": 0.00001,
        }
        with self.graph.as_default():
            # Import saved model from location 'loc' into local graph
            # 从指定路径加载模型到局部图中
            self.agents = [
                nfsp.NFSP(self.sess, idx, state_representation_size=info_state_size, num_actions=num_actions,
                          hidden_layers_sizes=hidden_layers_sizes,
                          reservoir_buffer_capacity=reservoir_buffer_capacity,
                          anticipatory_param=anticipatory_param,
                          **kwargs) for idx in range(num_players)
            ]
            saver = tf.train.Saver()
            saver.restore(self.sess, loc)
            # There are TWO options how to get activation operation:
            # 两种方式来调用运算或者参数
            # FROM SAVED COLLECTION:
            # self.activation = tf.get_collection('activation')[0]
            # BY NAME:
            # self.activation = self.graph.get_operation_by_name('activation_opt').outputs[0]

    def step(self, cur_player, data):
        """ Running the activation operation previously imported """
        # The 'x' corresponds to name of input placeholder
        return 1
        # return self.sess.run(self.activation, feed_dict={"x:0": data})


class ImportNFSP_ARM:
    """  Importing and running isolated TF graph """

    def __init__(self, info_state_size, num_actions, hidden_layers_sizes, loc):
        # Create local graph and use it in the session
        # origin nfsp
        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph)
        reservoir_buffer_capacity = 2e6
        anticipatory_param = 0.5
        num_players = 2
        kwargs = {
            "replay_buffer_capacity": 3000,
            "epsilon_decay_duration": 30000,
            "epsilon_start": 0.06,
            "epsilon_end": 0.00001,
        }
        with self.graph.as_default():
            # Import saved model from location 'loc' into local graph
            # 从指定路径加载模型到局部图中
            self.agents = [
                nfsp_arm.NFSP(self.sess, idx, info_state_size, num_actions, hidden_layers_sizes,
                              reservoir_buffer_capacity, anticipatory_param,
                              **kwargs) for idx in range(num_players)
            ]
            saver = tf.train.Saver()
            saver.restore(self.sess, loc)
            # There are TWO options how to get activation operation:
            # 两种方式来调用运算或者参数
            # FROM SAVED COLLECTION:
            # self.activation = tf.get_collection('activation')[0]
            # BY NAME:
            # self.activation = self.graph.get_operation_by_name('activation_opt').outputs[0]

    def step(self, cur_player, data):
        """ Running the activation operation previously imported """
        # The 'x' corresponds to name of input placeholder
        return 1
        # return self.sess.run(self.activation, feed_dict={"x:0": data})


def main(unused_argv):
    start_time = time.time()
    game = "leduc_poker"
    num_players = 2

    env_configs = {"players": num_players}
    env = rl_environment.Environment(game, **env_configs)
    # env = rl_environment.Environment(game)
    info_state_size = env.observation_spec()["info_state"][0]
    num_actions = env.action_spec()["num_actions"]

    hidden_layers_sizes = [int(l) for l in FLAGS.hidden_layers_sizes]

    # kwargs = {
    #     "replay_buffer_capacity": FLAGS.replay_buffer_capacity,
    #     "epsilon_decay_duration": FLAGS.num_train_episodes,
    #     "epsilon_start": 0.06,
    #     "epsilon_end": 0.001,
    # }

    # nfsp_arm
    kwargs = {
        "replay_buffer_capacity": FLAGS.replay_buffer_capacity,
        "epsilon_decay_duration": FLAGS.num_train_episodes,
        "epsilon_start": 0.06,
        "epsilon_end": 0.001,
    }

    hidden_layers_sizes_nfsp = [64]
    agents_nfsp = ImportNFSP(info_state_size, num_actions, hidden_layers_sizes_nfsp, './Model_NFSP/model.ckpt-13500000')
    expl_policies_avg_nfsp = NFSPPolicies(env, agents_nfsp.agents, nfsp.MODE.average_policy)
    expl_2 = exploitability.exploitability(env.game, expl_policies_avg_nfsp)

    agents = ImportNFSP_ARM(info_state_size, num_actions, hidden_layers_sizes, "./Model_NFSP_ARM_2/model.ckpt")
    expl_policies_avg = NFSPPolicies(env, agents.agents, nfsp_arm.MODE.average_policy)

    f_cum0 = open('./Model_NFSP_ARM_2/agent_0.pkl', 'rb')
    a0_cum_prob = pickle.load(f_cum0)
    f_cum0.close()

    f_cum1 = open('./Model_NFSP_ARM_2/agent_1.pkl', 'rb')
    a1_cum_prob = pickle.load(f_cum1)
    f_cum1.close()

    agents.agents[0].restore_cum_probs(a0_cum_prob)
    agents.agents[1].restore_cum_probs(a1_cum_prob)

    expl = exploitability.exploitability(env.game, expl_policies_avg)

    print(expl)
    print(expl_2)

    for i in range(100):
        r_mean = eval_against_random_bots_nfsp_arm(env, agents.agents, agents_nfsp.agents, 1000)
        logging.info("Mean episode rewards: %s, ", r_mean)
        with open('log/main_nfsp_arm_0_vs_nfsp1_v1.txt', 'a+') as f:
            f.write('{}\n'.format(r_mean[0]))
        with open('log/main_nfsp0_vs_nfsp_arm_1_v1.txt', 'a+') as f:
            f.write('{}\n'.format(r_mean[1]))


if __name__ == "__main__":
    app.run(main)
