from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import contextlib
import random
import enum
import numpy as np
import sonnet as snt
import tensorflow as tf

from open_spiel.python import rl_agent
from open_spiel.python.algorithms import dqn
import arm_tf

Transition = collections.namedtuple(
    "Transition", "info_state action_probs legal_actions_mask")

ILLEGAL_ACTION_LOGITS_PENALTY = -1e9

MODE = enum.Enum("mode", "best_response average_policy")


class NFSP(rl_agent.AbstractAgent):
    """NFSP_ARM Agent implementation in TensorFlow.

  See open_spiel/python/examples/nfsp.py for an usage example.
  """

    def __init__(self,
                 session,
                 player_id,
                 state_representation_size,
                 num_actions,
                 hidden_layers_sizes,
                 reservoir_buffer_capacity,
                 anticipatory_param,
                 batch_size=256,
                 rl_learning_rate=0.1,
                 sl_learning_rate=0.01,
                 min_buffer_size_to_learn=3000,
                 learn_every=64,
                 optimizer_str="adam",
                 **kwargs):
        """Initialize the `NFSP` agent."""
        self.player_id = player_id
        self._session = session
        self._num_actions = num_actions
        self._layer_sizes = hidden_layers_sizes + [num_actions]
        self._batch_size = batch_size
        self._learn_every = learn_every
        self._anticipatory_param = anticipatory_param
        self._min_buffer_size_to_learn = min_buffer_size_to_learn

        self._reservoir_buffer = ReservoirBuffer(reservoir_buffer_capacity)
        self._prev_timestep = None
        self._prev_action = None
        self.q_plus = 0.
        self.q_plus_all = None

        # Step counter to keep track of learning.
        self._step_counter = 0
        self._cum_probs = {}

        # Inner RL agent
        kwargs.update({
            "batch_size": batch_size,
            "learning_rate": rl_learning_rate,
            "learn_every": learn_every,
            "min_buffer_size_to_learn": min_buffer_size_to_learn,
            "optimizer_str": optimizer_str,
        })

        kwargs_rl = {
            "replay_buffer_capacity": int(3000),
            "epsilon_decay_duration": int(3e6),
            "epsilon_start": 0.06,
            "epsilon_end": 0.001,
        }

        self._rl_agent = arm_tf.ARM(session, player_id, state_representation_size,
                                           num_actions, hidden_layers_sizes, 1000, **kwargs_rl)

        # Keep track of the last training loss achieved in an update step.
        self._last_rl_loss_value = lambda: self._rl_agent.loss
        self._last_sl_loss_value = None

        # Placeholders.
        self._info_state_ph = tf.placeholder(
            shape=[None, state_representation_size],
            dtype=tf.float32,
            name="info_state_ph")

        self._action_probs_ph = tf.placeholder(
            shape=[None, num_actions], dtype=tf.float32, name="action_probs_ph")

        self._legal_actions_mask_ph = tf.placeholder(
            shape=[None, num_actions],
            dtype=tf.float32,
            name="legal_actions_mask_ph")

        # Average policy network.
        self._avg_network = snt.nets.MLP(output_sizes=self._layer_sizes)
        self._avg_policy = self._avg_network(self._info_state_ph)
        self._avg_policy_probs = tf.nn.softmax(self._avg_policy)

        # Loss
        self._loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(
                labels=tf.stop_gradient(self._action_probs_ph),
                logits=self._avg_policy))

        if optimizer_str == "adam":
            optimizer = tf.train.AdamOptimizer(learning_rate=sl_learning_rate)
        elif optimizer_str == "sgd":
            optimizer = tf.train.GradientDescentOptimizer(
                learning_rate=sl_learning_rate)
        else:
            raise ValueError("Not implemented. Choose from ['adam', 'sgd'].")

        self._learn_step = optimizer.minimize(self._loss)
        self._sample_episode_policy()

        self._br_count = 0
        self._ap_count = 0

    @contextlib.contextmanager
    def temp_mode_as(self, mode):
        """Context manager to temporarily overwrite the mode."""
        previous_mode = self._mode
        self._mode = mode
        yield
        self._mode = previous_mode

    def set_anti(self, anti):
        self._anticipatory_param = anti

    def _sample_episode_policy(self):
        if np.random.rand() < self._anticipatory_param:
            self._mode = MODE.best_response
        else:
            self._mode = MODE.average_policy

    def _act(self, info_state, legal_actions):
        info_state = np.reshape(info_state, [1, -1])
        action_values, action_probs = self._session.run(
            [self._avg_policy, self._avg_policy_probs],
            feed_dict={self._info_state_ph: info_state})

        self._last_action_values = action_values[0]
        # print("_act: {}, {}".format(action_values, action_values[0]))
        # Remove illegal actions, normalize probs
        probs = np.zeros(self._num_actions)
        probs[legal_actions] = action_probs[0][legal_actions]
        print(probs, sum(probs), action_values, action_probs)
        if sum(probs):
            probs /= sum(probs)
        action = np.random.choice(len(probs), p=probs)
        print(probs, action)
        return action, probs

    @property
    def mode(self):
        return self._mode

    @property
    def cum_prob(self):
        return self._cum_probs

    @property
    def loss(self):
        return (self._last_sl_loss_value, self._last_rl_loss_value())

    @property
    def rl_buffer_len(self):
        return len(self._rl_agent.replay_buffer)

    def restore_cum_probs(self, cum):
        self._cum_probs = cum

    def step(self, time_step, is_evaluation=False):
        """Returns the action to be taken and updates the Q-networks if needed.

    Args:
      time_step: an instance of rl_environment.TimeStep.
      is_evaluation: bool, whether this is a training or evaluation call.

    Returns:
      A `rl_agent.StepOutput` containing the action probs and chosen action.
    """
        if self._mode == MODE.best_response:
            agent_output = self._rl_agent.step(time_step, is_evaluation)
            #if agent_output is not None:
            if not is_evaluation and not time_step.last():
                self._add_transition(time_step, agent_output)

        elif self._mode == MODE.average_policy:
            # Act step: don't act at terminal info states.
            if not time_step.last():
                info_state = time_step.observations["info_state"][self.player_id]
                legal_actions = time_step.observations["legal_actions"][self.player_id]
                ret_probs = np.zeros(self._num_actions)
                info_state = np.reshape(info_state, [1, -1])
                # print("In ap: ", info_state)

                if str(info_state) in self._cum_probs:
                    action_values = self._cum_probs[str(info_state)]
                    legal_p_values = action_values[legal_actions]
                    p_values_sum = np.sum(legal_p_values)
                    if p_values_sum:
                        action_prob = legal_p_values / p_values_sum
                        for action in legal_actions:
                            ret_probs[action] = action_values[action] / p_values_sum

                        chosed_legal_action = np.random.choice(range(action_prob.shape[0]), p=action_prob.ravel())
                        action = legal_actions[chosed_legal_action]
                        agent_output = rl_agent.StepOutput(action=action, probs=ret_probs)
                    else:  # rl_output or 1/n? ignore
                        # print("In Else_2")
                        # print("in else_3")
                        # action, probs = self._act(info_state, legal_actions)
                        # agent_output = rl_agent.StepOutput(action=action, probs=probs)
                        agent_output = self._rl_agent.step(time_step, is_evaluation)
                        if agent_output is not None:
                            # print("Not none")
                            ### avg_strategy
                            info_state = time_step.observations["info_state"][self.player_id]
                            info_state = np.reshape(info_state, [1, -1])
                            probabilities = agent_output.probs
                            if str(info_state) not in self._cum_probs.keys():
                                self._cum_probs[str(info_state)] = probabilities
                            else:
                                for i in range(self._num_actions):
                                    self._cum_probs[str(info_state)][i] += probabilities[i] * 10
                else:
                    # print("In Else")
                    agent_output = self._rl_agent.step(time_step, is_evaluation=True)
                    if agent_output is not None:
                        # print("Not none")
                        ### avg_strategy
                        info_state = time_step.observations["info_state"][self.player_id]
                        info_state = np.reshape(info_state, [1, -1])
                        probabilities = agent_output.probs
                        if str(info_state) not in self._cum_probs.keys():
                            self._cum_probs[str(info_state)] = probabilities
                        else:
                            # print("In else_3")
                            for i in range(self._num_actions):
                                self._cum_probs[str(info_state)][i] += probabilities[i]

            if self._prev_timestep and not is_evaluation:
                self._rl_agent.add_transition(self._prev_timestep, self._prev_action, time_step)
        else:
            raise ValueError("Invalid mode ({})".format(self._mode))

        self.q_plus = self._rl_agent.q_plus
        self.q_plus_all = self._rl_agent.q_plus_all

        if not is_evaluation:
            self._step_counter += 1

            # if self._step_counter % self._learn_every == 0:
            #     self._last_sl_loss_value = self._learn()
                # If learn step not triggered by rl policy, learn.
                # if self._mode == MODE.average_policy:
            #         # self._rl_agent.learn()
            # if time_step.last():
            #     print("last: {}".format(self._rl_agent.step_counter))
            #     if self._rl_agent.step_counter > 3000:
            #         print("In nfsp_arm_learn")
            #         self._rl_agent.learn()
            #     else:
            #         print("In else")

            # Prepare for the next episode.
            if time_step.last():
                self._sample_episode_policy()
                self._prev_timestep = None
                self._prev_action = None
                return
            else:
                self._prev_timestep = time_step
                self._prev_action = agent_output.action

        return agent_output

    def _add_transition(self, time_step, agent_output):
        """Adds the new transition using `time_step` to the reservoir buffer.

    Transitions are in the form (time_step, agent_output.probs, legal_mask).

    Args:
      time_step: an instance of rl_environment.TimeStep.
      agent_output: an instance of rl_agent.StepOutput.
    """
        legal_actions = time_step.observations["legal_actions"][self.player_id]
        legal_actions_mask = np.zeros(self._num_actions)
        legal_actions_mask[legal_actions] = 1.0
        info_state = time_step.observations["info_state"][self.player_id]
        info_state_2 = np.reshape(info_state, [1, -1])
        # print(info_state, info_state_2)
        transition = Transition(
            info_state=info_state_2,  # (time_step.observations["info_state"][self.player_id][:])
            action_probs=agent_output.probs,
            legal_actions_mask=legal_actions_mask)
        self._reservoir_buffer.add(transition)

    def learn(self):
        # print("Min buffer: ", self._min_buffer_size_to_learn)
        if len(self._rl_agent.replay_buffer) < self._min_buffer_size_to_learn:
            return None
        # update rl_agent
        loss = self._rl_agent.learn()

        # update average policy
        for element in self._reservoir_buffer.data:
            info_state = element.info_state
            action_probs = element.action_probs
            legal_actions_mask = element.legal_actions_mask
            info_state_2 = np.reshape(info_state, [1, -1])
            # print(info_state, info_state_2)
            probabilities = action_probs
            if str(info_state) not in self._cum_probs.keys():
                self._cum_probs[str(info_state)] = probabilities
            else:
                for i in range(self._num_actions):
                    self._cum_probs[str(info_state)][i] += probabilities[i]
        self._reservoir_buffer.clear()

        return loss

    def param_out(self):
        print(self._rl_agent.step_counter, len(self._rl_agent.replay_buffer), self.mode, self._br_count, self._ap_count)

    def tag(self):
        return "nfsp_arm"


class ReservoirBuffer(object):
    """Allows uniform sampling over a stream of data.

  This class supports the storage of arbitrary elements, such as observation
  tensors, integer actions, etc.

  See https://en.wikipedia.org/wiki/Reservoir_sampling for more details.
  """

    def __init__(self, reservoir_buffer_capacity):
        self._reservoir_buffer_capacity = reservoir_buffer_capacity
        self.data = []
        self._add_calls = 0

    def add(self, element):
        """Potentially adds `element` to the reservoir buffer.

    Args:
      element: data to be added to the reservoir buffer.
    """
        if len(self.data) < self._reservoir_buffer_capacity:
            self.data.append(element)
        else:
            idx = np.random.randint(0, self._add_calls + 1)
            if idx < self._reservoir_buffer_capacity:
                self.data[idx] = element
        self._add_calls += 1

    def sample(self, num_samples):
        """Returns `num_samples` uniformly sampled from the buffer.

    Args:
      num_samples: `int`, number of samples to draw.

    Returns:
      An iterable over `num_samples` random elements of the buffer.

    Raises:
      ValueError: If there are less than `num_samples` elements in the buffer
    """
        if len(self.data) < num_samples:
            raise ValueError("{} elements could not be sampled from size {}".format(
                num_samples, len(self.data)))
        return random.sample(self.data, num_samples)

    def clear(self):
        self.data = []
        self._add_calls = 0

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        return iter(self.data)
