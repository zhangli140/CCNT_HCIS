"""Python Deep CFR example."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags
from absl import logging
import six

import tensorflow.compat.v1 as tf

from open_spiel.python import policy
#from open_spiel.python.algorithms import deep_cfr
from open_spiel.python.algorithms import expected_game_score
from open_spiel.python.algorithms import exploitability
import pyspiel
import deep_cfr

# Temporarily disable TF2 behavior until we update the code.
tf.disable_v2_behavior()

FLAGS = flags.FLAGS

flags.DEFINE_integer("num_iterations", 20000, "Number of iterations")
flags.DEFINE_integer("num_traversals", 80, "Number of traversals/games")
flags.DEFINE_string("game_name", "leduc_poker", "Name of the game")

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

def main(unused_argv):
  logging.info("Loading %s", FLAGS.game_name)
  game = pyspiel.load_game(FLAGS.game_name)
  with tf.Session(config=config) as sess:
    deep_cfr_solver = deep_cfr.DeepCFRSolver(
        sess,
        game,
        policy_network_layers=(128,64),
        advantage_network_layers=(128,64),
        num_iterations=FLAGS.num_iterations,
        num_traversals=FLAGS.num_traversals,
        learning_rate=1e-3,
        batch_size_advantage=128,
        batch_size_strategy=1024,
        memory_capacity=1e7,
        policy_network_train_steps=400,
        advantage_network_train_steps=300,
        reinitialize_advantage_networks=False)
    sess.run(tf.global_variables_initializer())
    _, advantage_losses, policy_loss = deep_cfr_solver.solve(game)
    for player, losses in six.iteritems(advantage_losses):
      logging.info("Advantage for player %d: %s", player,
                   losses[:2] + ["..."] + losses[-2:])
      logging.info("Advantage Buffer Size for player %s: '%s'", player,
                   len(deep_cfr_solver.advantage_buffers[player]))
    logging.info("Strategy Buffer Size: '%s'",
                 len(deep_cfr_solver.strategy_buffer))
    logging.info("Final policy loss: '%s'", policy_loss)

    average_policy = policy.tabular_policy_from_callable(
        game, deep_cfr_solver.action_probabilities)

    conv = exploitability.exploitability(game, average_policy)
    logging.info("Deep CFR in '%s' - Exploitability: %s", FLAGS.game_name, conv)

    average_policy_values = expected_game_score.policy_value(
        game.new_initial_state(), [average_policy] * 2)
    print("Computed player 0 value: {}".format(average_policy_values[0]))
    print("Expected player 0 value: {}".format(-1 / 18))
    print("Computed player 1 value: {}".format(average_policy_values[1]))
    print("Expected player 1 value: {}".format(1 / 18))


if __name__ == "__main__":
  app.run(main)


