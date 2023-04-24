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
import arm_tf
import nfsp_arm
import nfsp

import matplotlib.pyplot as plt

FLAGS = flags.FLAGS

flags.DEFINE_integer("num_train_episodes", int(15e6),
                     "Number of training episodes.")
flags.DEFINE_integer("eval_every", 10000,
                     "Episode frequency at which the agents are evaluated.")
flags.DEFINE_list("hidden_layers_sizes", [
    64
], "Number of hidden units in the avg-net and Q-net.")
flags.DEFINE_integer("replay_buffer_capacity", int(200000),
                     "Size of the replay buffer.")
flags.DEFINE_integer("reservoir_buffer_capacity", int(2e6),
                     "Size of the reservoir buffer.")
flags.DEFINE_float("anticipatory_param", 0.5,
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

        with self._policies[cur_player].temp_mode_as(self._mode):
            p = self._policies[cur_player].step(info_state, is_evaluation=True).probs
            prob_dict = {action: p[action] for action in legal_actions}
        return prob_dict


def main(unused_argv):
    game = "leduc_poker"
    num_players = 2

    env_configs = {"players": num_players}
    env = rl_environment.Environment(game, **env_configs)
    info_state_size = env.observation_spec()["info_state"][0]
    num_actions = env.action_spec()["num_actions"]

    hidden_layers_sizes = [int(l) for l in FLAGS.hidden_layers_sizes]

    kwargs = {
        "replay_buffer_capacity": FLAGS.replay_buffer_capacity,
        "epsilon_decay_duration": FLAGS.num_train_episodes,
        "epsilon_start": 0.06,
        "epsilon_end": 0.00001,
    }

    exploit = []

    with tf.compat.v1.Session() as sess:
        # pylint: disable=g-complex-comprehension
        agents = [
            nfsp.NFSP(sess, idx, info_state_size, num_actions, hidden_layers_sizes,
                      FLAGS.reservoir_buffer_capacity, FLAGS.anticipatory_param,
                      **kwargs) for idx in range(num_players)
        ]
        # agents_tst = [arm_tf_tst.ARM(sess, idx, info_state_size, num_actions, hidden_layers_sizes, 1000,
        #                      **kwargs) for idx in range(num_players)]
        # agents = [arm_tf.ARM(sess, idx, info_state_size, num_actions, hidden_layers_sizes, 1000,
        #                      **kwargs) for idx in range(num_players)]

        expl_policies_avg = NFSPPolicies(env, agents, nfsp.MODE.average_policy)

        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        # print(FLAGS.num_train_episodes)
        for ep in range(FLAGS.num_train_episodes):
            if (ep + 1) % 500000 == 0:
                saver.save(sess, "./Model_NFSP/model.ckpt", global_step=ep+1)

            if (ep + 1) % FLAGS.eval_every == 0:
                losses = [agent.loss for agent in agents]
                print("Losses: {}".format(losses))
                expl = exploitability.exploitability(env.game, expl_policies_avg)
                print("{} Exploitability AVG {}".format(ep + 1, expl))
                print("_____________________________________________")
                exploit.append(expl)
                with open("log/loss_nfsp.txt", 'a+') as f:
                    f.write("{}\n".format(losses))
                with open("log/ex_nfsp.txt", "a+") as f:
                    f.write("{}\n".format(expl))

            time_step = env.reset()
            while not time_step.last():
                player_id = time_step.observations["current_player"]
                agent_output = agents[player_id].step(time_step)
                action_list = [agent_output.action]
                time_step = env.step(action_list)

            # Episode is over, step all agents with final info state.
            for agent in agents:
                agent.step(time_step)

        saver.save(sess, "./Model_NFSP/model.ckpt")

    plt.plot(exploit)
    plt.show()


if __name__ == "__main__":
    app.run(main)
