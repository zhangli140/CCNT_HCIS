from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import random
import numpy as np
import sonnet as snt
import tensorflow as tf
import enum
import torch

from open_spiel.python import rl_agent
from buffer import ReplayBuffer

Transition = collections.namedtuple(
    "Transition",
    "info_state action reward next_info_state is_final_step legal_actions_mask")

ILLEGAL_ACTION_LOGITS_PENALTY = -1e9
N_STEP_SIZE = 4
GAMMA = 0.90
MODE = enum.Enum("mode", "best_response average_policy")


class ARM(rl_agent.AbstractAgent):

    def __init__(self,
                 session,
                 player_id,
                 state_representation_size,
                 num_actions,
                 hidden_layers_sizes,
                 iterations=1000,
                 replay_buffer_capacity=3000,
                 batch_size=32,
                 replay_buffer_class=ReplayBuffer,
                 learning_rate=0.0008,
                 update_target_network_every=1000,
                 learn_every=10,
                 discount_factor=1.0,
                 min_buffer_size_to_learn=1000,
                 epsilon_start=1.0,
                 epsilon_end=0.1,
                 epsilon_decay_duration=int(1e6),
                 optimizer_str="adam",
                 loss_str="mse"):
        """Initialize the agent."""
        self.player_id = player_id
        self._session = session
        self._num_actions = num_actions
        self._layer_sizes = hidden_layers_sizes + [num_actions + 1]
        self._batch_size = batch_size
        self._update_target_network_every = update_target_network_every
        self._learn_every = learn_every
        self._min_buffer_size_to_learn = 2000  # min_buffer_size_to_learn
        self._discount_factor = discount_factor

        self._epsilon_start = epsilon_start
        self._epsilon_end = epsilon_end
        self._epsilon_decay_duration = epsilon_decay_duration

        # TODO(author6) Allow for optional replay buffer config.
        self._replay_buffer = replay_buffer_class(replay_buffer_capacity)
        self._replay_buffer_capacity = replay_buffer_capacity
        self._prev_timestep = None
        self._prev_action = None

        # Step counter to keep track of learning, eps decay and target network.
        self._step_counter = 0

        # Iterations while training
        self.iteration = iterations

        # Keep track of the last training loss achieved in an update step.
        self._last_loss_value = None

        # Create required TensorFlow placeholders to perform the Q-network updates.
        self._info_state_ph = tf.compat.v1.placeholder(
            shape=[None, state_representation_size],
            dtype=tf.float32,
            name="info_state_ph")
        self._action_ph = tf.compat.v1.placeholder(
            shape=[None], dtype=tf.int32, name="action_ph")
        self._reward_ph = tf.compat.v1.placeholder(
            shape=[None], dtype=tf.float32, name="reward_ph")
        self._is_final_step_ph = tf.compat.v1.placeholder(
            shape=[None], dtype=tf.float32, name="is_final_step_ph")
        self._next_info_state_ph = tf.compat.v1.placeholder(
            shape=[None, state_representation_size],
            dtype=tf.float32,
            name="next_info_state_ph")
        self._legal_actions_mask_ph = tf.compat.v1.placeholder(
            shape=[None, num_actions],
            dtype=tf.float32,
            name="legal_actions_mask_ph")
        # placeholder for sampled_indices
        # self._sampled_indices_ph = tf.placeholder(
        #     shape=[batch_size],
        #     dtype=tf.int64,
        #     name="sampled_indices")
        self._n_step_ph = tf.placeholder(shape=[None], dtype=tf.float32, name="n_step")
        self._mb_est_rew_ph = tf.placeholder(shape=[None], dtype=tf.float32,
                                             name="mb_est_rew")
        self._tar_v_ph = tf.placeholder(shape=[None], dtype=tf.float32, name="tar_v")
        self._tar_q_ph = tf.placeholder(shape=[None], dtype=tf.float32, name="tar_q")
        print("In init: {}, {}".format(self._tar_q_ph, self._tar_q_ph))
        self.tau = 0.01
        self.q_plus_weight = 1
        self.clip_value = True
        self._mode = MODE.best_response

        self._q_network = snt.nets.MLP(output_sizes=self._layer_sizes)
        self._q_values = self._q_network(self._info_state_ph)
        self._target_q_network = snt.nets.MLP(output_sizes=self._layer_sizes)
        self._target_q_values = self._target_q_network(self._next_info_state_ph)

        # Stop gradient to prevent updates to the target network while learning
        # self._target_q_values = tf.stop_gradient(self._target_q_values)

        self._update_target_network = self._create_target_network_update_op(
            self._q_network, self._target_q_network)

        # Create the loss operations.
        # Sum a large negative constant to illegal action logits before taking the
        # max. This prevents illegal action values from being considered as target.
        illegal_actions = 1 - self._legal_actions_mask_ph
        illegal_logits = illegal_actions * ILLEGAL_ACTION_LOGITS_PENALTY

        # max_next_q = tf.reduce_max(
        #     tf.math.add(tf.stop_gradient(self._target_q_values), illegal_logits),
        #     axis=-1)
        # target = (
        #         self._reward_ph +
        #         (1 - self._is_final_step_ph) * self._discount_factor * max_next_q)

        # action_indices = tf.stack(
        #     [tf.range(tf.shape(self._q_values)[0]), self._action_ph], axis=-1)
        # predictions = tf.gather_nd(self._q_values, action_indices)

        if loss_str == "mse":
            loss_class = tf.compat.v1.losses.mean_squared_error
        elif loss_str == "huber":  # SmoothL1Loss
            loss_class = tf.compat.v1.losses.huber_loss
        else:
            raise ValueError("Not implemented, choose from 'mse', 'huber'.")

        # self.v = 0.
        # self.q = 0.
        # self.tar_v = 0.
        # self.tar_q = 0.
        self._sampled_indices = np.zeros((self._batch_size, 1), dtype=int)
        self.q_plus = None
        self.q_plus_all = None
        self.v, self.q, self.tar_v, self.tar_q = self.__compute_losses(self._sampled_indices)
        self.v_loss = loss_class(labels=self.tar_v, predictions=self.v)
        self.q_loss = loss_class(labels=self.tar_q, predictions=self.q)
        print("In init: {}, {}, {}, {}".format(self.v, self.q, self.tar_v, self.tar_q))
        print(self.v_loss, self.q_loss)
        print(self.q_loss + self.v_loss)

        self._loss = tf.reduce_mean(self.v_loss + self.q_loss)
        print(self._loss)

        if optimizer_str == "adam":
            optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)
        elif optimizer_str == "sgd":
            optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=learning_rate)
        else:
            raise ValueError("Not implemented, choose from 'adam' and 'sgd'.")

        if callable(self._loss):
            print("callable")
        else:
            print("Not callable")
        self._learn_step = optimizer.minimize(self._loss)

    def step(self, time_step, is_evaluation=False, add_transition_record=True):
        """Returns the action to be taken and updates the network if needed.

    Args:
      time_step: an instance of rl_environment.TimeStep.
      is_evaluation: bool, whether this is a training or evaluation call.
      add_transition_record: Whether to add to the replay buffer on this step.

    Returns:
      A `rl_agent.StepOutput` containing the action probs and chosen action.
    """
        # Act step: don't act at terminal info states or if its not our turn.
        if (not time_step.last()) and (
                time_step.is_simultaneous_move() or
                self.player_id == time_step.current_player()):
            info_state = time_step.observations["info_state"][self.player_id]
            legal_actions = time_step.observations["legal_actions"][self.player_id]
            # ---------Get Action (Policy)------------#
            epsilon = self._get_epsilon(is_evaluation)
            action, probs = self._action_policy(info_state, legal_actions, epsilon)
            # action, probs = self._epsilon_greedy(info_state, legal_actions, epsilon)
        else:
            action = None
            probs = []

        # Don't mess up with the state during evaluation.
        if not is_evaluation:
            if self.prev_timestep:
                self._step_counter += 1

            # if self._step_counter % self._learn_every == 0:
            #     self._last_loss_value = self.learn()

            # if self._step_counter % self._update_target_network_every == 0:
            #     self._session.run(self._update_target_network)

            if self._prev_timestep and add_transition_record:
                # We may omit record adding here if it's done elsewhere.
                self.add_transition(self._prev_timestep, self._prev_action, time_step)

            if time_step.last():  # prepare for the next episode.
                # if self._step_counter > self._replay_buffer_capacity:  # count of steps is larger than set length
                #     # print("In arm_tf_learn")
                #     self._last_loss_value = self.learn()
                #     # print("player: {}, step_counter: {}, loss: {}".format(self.player_id, self.step_counter, self._last_loss_value))
                #     self._step_counter = 0
                #     self._replay_buffer.clear()

                self._prev_timestep = None
                self._prev_action = None
                return
            else:
                self._prev_timestep = time_step
                self._prev_action = action

        return rl_agent.StepOutput(action=action, probs=probs)

    def add_transition(self, prev_time_step, prev_action, time_step):
        """Adds the new transition using `time_step` to the replay buffer.

    Adds the transition from `self._prev_timestep` to `time_step` by
    `self._prev_action`.

    Args:
      prev_time_step: prev ts, an instance of rl_environment.TimeStep.
      prev_action: int, action taken at `prev_time_step`.
      time_step: current ts, an instance of rl_environment.TimeStep.
    """
        assert prev_time_step is not None
        legal_actions = (
            prev_time_step.observations["legal_actions"][self.player_id])
        legal_actions_mask = np.zeros(self._num_actions)
        legal_actions_mask[legal_actions] = 1.0
        # transition = Transition(
        #     info_state=(
        #         prev_time_step.observations["info_state"][self.player_id][:]),
        #     action=prev_action,
        #     reward=time_step.rewards[self.player_id],
        #     next_info_state=time_step.observations["info_state"][self.player_id][:],
        #     is_final_step=float(time_step.last()),
        #     legal_actions_mask=legal_actions_mask)
        # print(time_step.last(), float(time_step.last()))
        self._replay_buffer.add(
            prev_time_step.observations["info_state"][self.player_id][:],
            time_step.observations["info_state"][self.player_id][:],
            prev_action,
            time_step.rewards[self.player_id],
            time_step.last(),
            legal_actions_mask)

    def _create_target_network_update_op(self, q_network, target_q_network):
        """Create TF ops copying the params of the network to the target network.

    Args:
      q_network: `snt.AbstractModule`. Values are copied from this network.
      target_q_network: `snt.AbstractModule`. Values are copied to this network.

    Returns:
      A `tf.Operation` that updates the variables of the target.
    """
        variables = q_network.get_variables()
        target_variables = target_q_network.get_variables()
        # problem
        return tf.group([
            tf.assign(target_v, target_v + self.tau * (v - target_v))  # same as original arm
            for (target_v, v) in zip(target_variables, variables)
        ])

    def _epsilon_greedy(self, info_state, legal_actions, epsilon):
        """Returns a valid epsilon-greedy action and valid action probs.

    Action probabilities are given by a softmax over legal q-values.

    Args:
      info_state: hashable representation of the information state.
      legal_actions: list of legal actions at `info_state`.
      epsilon: float, probability of taking an exploratory action.

    Returns:
      A valid epsilon-greedy action and valid action probabilities.
    """
        probs = np.zeros(self._num_actions)
        if np.random.rand() < epsilon:
            action = np.random.choice(legal_actions)
            probs[legal_actions] = 1.0 / len(legal_actions)
        else:
            info_state = np.reshape(info_state, [1, -1])
            q_values = self._session.run(
                self._q_values, feed_dict={self._info_state_ph: info_state})[0]
            legal_q_values = q_values[legal_actions]
            action = legal_actions[np.argmax(legal_q_values)]
            probs[action] = 1.0
        return action, probs

    def _get_epsilon(self, is_evaluation, power=1.0):
        """Returns the evaluation or decayed epsilon value."""
        if is_evaluation:
            return 0.0
        decay_steps = min(self._step_counter, self._epsilon_decay_duration)
        decayed_epsilon = (
                self._epsilon_end + (self._epsilon_start - self._epsilon_end) *
                (1 - decay_steps / self._epsilon_decay_duration) ** power)
        return decayed_epsilon

    def learn(self):
        """Compute the loss on sampled transitions and perform a network update.

    If there are not enough elements in the buffer, no loss is computed and
    `None` is returned instead.

    Returns:
      The average loss obtained on this batch of transitions or `None`.
    """

        if (len(self._replay_buffer) < self._batch_size or
                len(self._replay_buffer) < self._min_buffer_size_to_learn):
            return None
        # print(len(self._replay_buffer), self._min_buffer_size_to_learn)

        cum_v_loss = 0.
        cum_q_loss = 0.

        # ----------------   compute all targets (not necessary)---------------#
        # vectorize may have problem.
        self._replay_buffer.vectorize(frame_buffer=0, n_step_size=N_STEP_SIZE, gamma=GAMMA)
        # print(self._replay_buffer.idcs, self._replay_buffer.n_step)
        # change to getting all target values first
        all_tar_v, all_tar_q, q_plus = self.__compute_targets(self._sampled_indices)
        # if self._replay_buffer.n_step.shape == (0,):
        #     return
        all_v_mb, all_q_mb = self._session.run([all_tar_v, all_tar_q],
                                               feed_dict={self._info_state_ph: self._replay_buffer.obs,
                                                          self._n_step_ph: self._replay_buffer.n_step[self._replay_buffer.idcs],
                                                          self._action_ph: self._replay_buffer.actions})

        # ----------------every iteration, sample and compute loss-------------#
        for i_iter in range(self.iteration):
            # Sample (get states & ...)
            mb_idcs = np.random.choice(len(self._replay_buffer), self._batch_size)
            mb_info_state, _, mb_actions, *_ = self._replay_buffer[mb_idcs]  # problem

            mb_est_rew_w = self._replay_buffer.est_rew_weights[self._replay_buffer.idcs[mb_idcs]]  # problem
            mb_est_non_zero = np.nonzero(mb_est_rew_w)
            n_step = self._replay_buffer.n_step[mb_idcs]

            if len(mb_est_non_zero[0]):  # the array is not empty
                mb_est_non_zero = np.squeeze(mb_est_non_zero)
                mb_est_rew_idcs = (self._replay_buffer.idcs[mb_idcs][mb_est_non_zero] +
                                   self._replay_buffer.n_step_size).reshape(-1)

                # mb_v_prime_obs = next_info_state_ph
                mb_v_prime_obs, _, _, *_ = self._replay_buffer[mb_est_rew_idcs]
            else:
                mb_v_prime_obs = np.zeros((32, mb_info_state[0].size))

            # if self.player_id == 0:
            #     print("In Learn Test Start: {}#################".format(i_iter))
            #     print(self.player_id)
            #     print(mb_info_state.shape)
            #     print(mb_actions.shape)
            #     print(mb_v_prime_obs.shape)
            #     print(n_step.shape)
            #     print(mb_est_rew_w)
            #     print(mb_est_non_zero)
            #     print("In
            #     Test End: {}###################".format(i_iter))
            tar_v_mb = all_v_mb[mb_idcs]
            tar_q_mb = all_q_mb[mb_idcs]
            # print("In iteration: {}".format(i_iter))
            # print(tar_v_mb.shape, tar_q_mb.shape)

            loss, _, v_loss, q_loss = self._session.run(
                [self._loss, self._learn_step, self.v_loss, self.q_loss],
                feed_dict={
                    self._info_state_ph: mb_info_state,
                    self._action_ph: mb_actions,
                    self._next_info_state_ph: mb_v_prime_obs,
                    self._mb_est_rew_ph: mb_est_rew_w,
                    self._tar_q_ph: tar_q_mb,
                    self._tar_v_ph: tar_v_mb
                })

            self._session.run(self._update_target_network)

            cum_q_loss += q_loss
            cum_v_loss += v_loss

            if (i_iter + 1) % (self.iteration / 10) == 0.:
                # print loss
                mean_v_loss = (cum_v_loss / int(self.iteration / 10))
                mean_q_loss = (cum_q_loss / int(self.iteration / 10))
                print("interation: {}, v_loss: {:.6f}, q_loss: {:.6f}".format(
                    i_iter + 1, mean_v_loss, mean_q_loss), end='\r')
                if (i_iter + 1) == self.iteration:
                    cum_v_loss = mean_v_loss
                    cum_q_loss = mean_q_loss
                else:
                    cum_v_loss = 0.0
                    cum_q_loss = 0.0

        # transitions = self._replay_buffer.sample(self._batch_size)
        # info_states = [t.info_state for t in transitions]
        # actions = [t.action for t in transitions]
        # rewards = [t.reward for t in transitions]
        # next_info_states = [t.next_info_state for t in transitions]
        # are_final_steps = [t.is_final_step for t in transitions]
        # legal_actions_mask = [t.legal_actions_mask for t in transitions]
        # loss, _ = self._session.run(
        #     [self._loss, self._learn_step],
        #     feed_dict={
        #         self._info_state_ph: info_states,
        #         self._action_ph: actions,
        #         self._reward_ph: rewards,
        #         self._is_final_step_ph: are_final_steps,
        #         self._next_info_state_ph: next_info_states,
        #         self._legal_actions_mask_ph: legal_actions_mask,
        #     })
        self._last_loss_value = [cum_v_loss, cum_q_loss]
        self.replay_buffer.clear()
        return [cum_v_loss, cum_q_loss]

    # -----------------------------ARM Functions -------------------------------#
    def __compute_losses(self, sampled_indices):
        illegal_actions = 1 - self._legal_actions_mask_ph
        illegal_logits = illegal_actions * ILLEGAL_ACTION_LOGITS_PENALTY
        # preprocessed_target_q = tf.math.add(tf.stop_gradient(self._target_q_values), illegal_logits)

        # tar_v, tar_q, q_plus = self.__compute_targets(sampled_indices)
        # print("In compute_loss: {}, {}".format(tar_q, tar_v))

        # val_est_mb = tf.zeros([])
        # mb_est_rew_w = self._replay_buffer.est_rew_weights[self._replay_buffer.idcs[sampled_indices]]  # problem
        # mb_est_rew_w = np.ones((self._batch_size, 1))
        nonzero_idx = tf.where(self._mb_est_rew_ph)
        count_nzero = tf.count_nonzero(self._mb_est_rew_ph)
        const_zero = tf.constant(0, dtype=tf.int64)

        def idx_nonezero():
            t_v = self._tar_v_ph
            t_q = self._tar_q_ph
            val_est = tf.stop_gradient(self._target_q_values)[:,
                      self._num_actions] * self._replay_buffer.gamma ** self._replay_buffer.n_step_size
            # print("shape: {}".format(val_est))
            val_est_mb = tf.zeros_like(self._tar_v_ph)
            print(nonzero_idx, val_est_mb, val_est)
            val_est_mb = tf.tensor_scatter_nd_add(val_est_mb, nonzero_idx, val_est)
            t_v = t_v + val_est_mb
            t_q = t_q + val_est_mb
            return t_v, t_q

        def idx_zero():
            t_v = self._tar_v_ph
            t_q = self._tar_q_ph
            return t_v, t_q

        tar_v, tar_q = tf.cond(tf.equal(count_nzero, const_zero), idx_zero, idx_nonezero)
        # mb_est_non_zero = tf.gather_nd(self._mb_est_rew_ph, nonzero_idx)
        # mb_est_non_zero = np.squeeze(np.nonzero(mb_est_rew_w))
        # if nonzero_idx is not None:  # the array is not empty

        # for i in range(len(mb_est_non_zero)):
        #     tf.gather()
        #     tf.assign(test[i], val_est[i])
        # val_est_mb[mb_est_non_zero[i]][0] = val_est[i]

        # mb_obs = self._info_state_ph
        # mb_actions = np.expand_dims(self._action_ph, 1)

        # compute current v and q values
        mb_values = self.q_values
        mb_v = mb_values[:, self._num_actions]
        mb_q = mb_values[:, :self._num_actions]
        print("Shape2: {}, {}, {}".format(mb_v.shape, mb_q.shape, self._action_ph.shape))
        action_indices = tf.stack(
            [tf.range(tf.shape(mb_q)[0]), self._action_ph], axis=-1)
        mb_q_t = tf.gather_nd(mb_q, action_indices)
        # action_one_hot = tf.one_hot(self._action_ph, 3)
        # mb_q_t = tf.diag_part(tf.matmul(mb_q, tf.transpose(action_one_hot)))
        # mb_q = tf.gather(mb_q, self._action_ph, batch_dims=1)  # problem
        print(mb_q_t)
        # add value estimate onto target values
        v = mb_v
        q = mb_q_t
        print("In losses: {}, {}, {}, {}".format(v, q, tar_v, tar_q))
        return v, q, tar_v, tar_q

    def __compute_targets(self, sampled_indices):
        # first_batch = not self._step_counter
        first_batch = self._step_counter

        # precompute all v and q target values
        # evs = tf.zeros([])
        # cfv = tf.zeros([])
        zero = tf.constant(0, dtype=tf.int32)

        def idx_nonezero_t():
            q_values = self._q_values
            # q_values_t = torch.from_numpy(q_values)
            q_evs = q_values[:, self._num_actions]
            q_cfv = q_values[:, :self._num_actions]
            # print("In compute target_1: {}, {}, {}".format(tf.rank(self._action_ph), q_evs, self._action_ph))
            action_indices = tf.stack(
                [tf.range(tf.shape(q_cfv)[0]), self._action_ph], axis=-1)
            q_cfv_t = tf.gather_nd(q_cfv, action_indices)
            # action_one_hot = tf.one_hot(self._action_ph, 3)
            # print(action_one_hot, q_cfv)
            # q_cfv_t = tf.diag_part(tf.matmul(q_cfv, tf.transpose(action_one_hot)))
            # print(mul_t)
            # q_cfv = tf.gather_nd(q_cfv, self._action_ph, batch_dims=1)  # problem
            evs = q_evs
            cfv = q_cfv_t
            # print(evs, cfv)
            # compute advantage value and clip to 0
            # print("In compute target_2: {}, {}".format(cfv, evs))
            q_plus_t = cfv - evs  # q_plus_t is going to be written
            # with open("log/q_plus.txt", 'a+') as f:
            #    f.write("{}\n".format(q_plus_t.eval()))
            # print("q_plus: {}".format(q_plus_t))
            if self.clip_value:
                q_plus_t = tf.clip_by_value(q_plus_t, clip_value_min=0, clip_value_max=1e999999)
            q_plus = q_plus_t
            n_step = self._n_step_ph

            v_t = n_step
            q_t = q_plus * self.q_plus_weight + n_step
            return v_t, q_t, q_plus

        def idx_zero_t():
            n_step = self._n_step_ph
            q_plus = tf.zeros_like(n_step)

            v_t = n_step
            q_t = n_step
            return v_t, q_t, q_plus

        v_tar, q_tar, q_plus = tf.cond(tf.equal(first_batch, zero), idx_zero_t, idx_nonezero_t)

        # if first_batch:
        # q_plus = np.zeros(self._batch_size)  # problem?

        # else:
        # compute q and v values of last iteration (problem)
        # obs = info_state
        # actions = actions
        # q_values = self._session.run(
        #     self._q_values, feed_dict={self._info_state_ph: obs})[0]  #maybe self._q_values is enough

        # print(sampled_indices)
        # n_step = self._replay_buffer.n_step[sampled_indices]  # problem
        # sampled_indices = self._sampled_indices_ph.eval()
        # print("sampled_indices: {}, {}".format(self._sampled_indices_ph, sampled_indices))
        # n_step = [self._replay_buffer.n_step[i] for i in sampled_indices]
        # n_step = np.zeros((self._batch_size, 1))
        # print("In target: {}, {}".format(v_tar, q_tar))
        return v_tar, q_tar, q_plus

    # def __sample_mini_batch(self, replay_buffer, v_tar, q_tar):
    #     # sample random batch from replay buffer indices
    #     mb_idcs = np.random.choice(len(replay_buffer), self._batch_size)
    #     mb_obs, _, mb_actions, *_ = replay_buffer[mb_idcs]  # problem
    #
    #     # initialize value estimate
    #     val_est_mb = np.zeros((self._batch_size, 1))
    #
    #     # compute value estimate for non terminal nodes
    #     mb_est_rew_w = replay_buffer.est_rew_weights[replay_buffer.idcs[mb_idcs]]  # problem
    #     mb_est_non_zero = np.squeeze(np.nonzero(mb_est_rew_w))
    #     if mb_est_non_zero:  # the array is not empty
    #         mb_est_rew_idcs = (replay_buffer.idcs[mb_idcs][mb_est_non_zero] +
    #                            replay_buffer.n_step_size).reshape(-1)
    #         # mb_v_prime_obs = replay_buffer.vec_obs[mb_est_rew_idcs]
    #         mb_v_prime_obs, _, _, *_ = replay_buffer[mb_est_rew_idcs]
    #         # print('Input2: {}, {}'.format(mb_est_rew_idcs.shape, replay_buffer.idcs[mb_idcs][mb_est_non_zero].shape))
    #         # print('Input3: {}, {}'.format(mb_v_prime_obs.shape, mb_est_rew_idcs.shape))
    #         mb_v_prime_actions = replay_buffer.vec_actions[mb_est_rew_idcs].astype(np.int64)
    #         # print('Input4: {}'.format(mb_v_prime_actions.shape))
    #         val_est = self._session.run(
    #             self._target_q_values, feed_dict={self._info_state_ph: mb_v_prime_obs})[:, self._num_actions]
    #         val_est = val_est * replay_buffer.gamma ** replay_buffer.n_step_size
    #         for i in range(len(mb_est_non_zero)):
    #             val_est_mb = val_est_mb + val_est[mb_est_non_zero[i]]
    #
    #     mb_obs = mb_obs
    #     mb_actions = np.expand_dims(mb_actions, 1)
    #
    #     # compute current v and q values
    #     mb_values = self._session.run(
    #         self._q_values, feed_dict={self._info_state_ph: mb_obs})
    #     mb_v = mb_values[:, self._num_actions]
    #     mb_q = mb_values[:, :self._num_actions]
    #     mb_q = self._session.run(tf.gather(mb_q, 1, mb_actions))  # problem
    #     # add value estimate onto target values
    #     mb_v_tar = v_tar[mb_idcs] + val_est_mb
    #     mb_q_tar = q_tar[mb_idcs] + val_est_mb
    #     return mb_v, mb_v_tar, mb_q, mb_q_tar

    def _action_policy(self, info_state, legal_actions, epsilon):
        probabilities = np.zeros(self._num_actions)
        info_state = np.reshape(info_state, [1, -1])
        q_values = self._session.run(
            self._q_values, feed_dict={self._info_state_ph: info_state})[0]

        expected_values = q_values[self._num_actions]  # index num_action is for value (different from original arm)
        cf_values = q_values[:self._num_actions]
        # print(q_values)
        # print("In action_policy: {}, {}, {}".format(expected_values, cf_values, cf_values - expected_values))
        action_values = np.clip(cf_values - expected_values, a_min=0, a_max=1e99999)
        # action_values = tf.clip_by_value(cf_values - expected_values, clip_value_min=0, clip_value_max=1e99999)
        legal_q_values = action_values[legal_actions]
        # print(action_values, legal_q_values, legal_actions)
        q_values_sum = np.sum(legal_q_values)
        if q_values_sum:  # Not zero
            action_prob = legal_q_values / q_values_sum
            for action in legal_actions:
                probabilities[action] = action_values[action] / q_values_sum
            # In DQN action_prob is set 1 for chosen action, but here not set 1
        else:
            action_dim = len(legal_actions)
            action_prob = np.ones(legal_q_values.shape)
            action_prob = action_prob / action_dim
            probabilities[legal_actions] = 1.0 / len(legal_actions)
        # action = legal_actions[np.argmax(action_prob)]  # problem: maybe legal_q_values is enough
        chosed_legal_action = np.random.choice(range(action_prob.shape[0]), p=action_prob.ravel())
        action = legal_actions[chosed_legal_action]
        self.q_plus = action_values[action]
        self.q_plus_all = action_values
        # print(action, action_prob, probabilities, chosed_legal_action)
        return action, probabilities

    def temp_mode_as(self, mode):
        """Context manager to temporarily overwrite the mode."""
        previous_mode = self._mode
        self._mode = mode
        yield
        self._mode = previous_mode

    def tag(self):
        return "arm"

    # -----------------------------ARM Functions -------------------------------#

    @property
    def q_values(self):
        return self._q_values

    @property
    def replay_buffer(self):
        return self._replay_buffer

    @property
    def info_state_ph(self):
        return self._info_state_ph

    @property
    def loss(self):
        return self._last_loss_value

    @property
    def prev_timestep(self):
        return self._prev_timestep

    @property
    def prev_action(self):
        return self._prev_action

    @property
    def step_counter(self):
        return self._step_counter
