import os
import torch
import numpy as np
import torch.nn.functional as F
from torch.optim import Adam
from torch.distributions import Categorical
from sacd.utils import soft_update, hard_update
from sacd.model import GaussianPolicy, QNetwork, DeterministicPolicy
from pommerman import agents


class SAC(agents.BaseAgent):
    def __init__(self, num_inputs, action_space, args):
        super(SAC, self).__init__()
        self.num_inputs = num_inputs
        self.action_space = action_space
        self.gamma = args.gamma
        self.tau = args.tau
        self.alpha = args.alpha

        self.policy_type = args.policy
        self.target_update_interval = args.target_update_interval
        self.automatic_entropy_tuning = args.automatic_entropy_tuning

        self.device = torch.device("cuda" if args.cuda else "cpu") 

        self.critic = QNetwork(num_inputs, action_space.n, args.hidden_size).to(device=self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr=args.lr)

        self.critic_target = QNetwork(num_inputs, action_space.n, args.hidden_size).to(self.device)
        hard_update(self.critic_target, self.critic)

        if self.policy_type != "Gaussian":
            # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
            if self.automatic_entropy_tuning:
                self.target_entropy = -np.log((1.0 / action_space.n)) * 0.98
                self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
                self.alpha_optim = Adam([self.log_alpha], lr=args.lr)
            self.policy = DeterministicPolicy(num_inputs, action_space.n, args.hidden_size, action_space).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)

        else:
            self.alpha = 0
            self.automatic_entropy_tuning = False
            self.policy = GaussianPolicy(num_inputs, action_space.n, args.hidden_size, action_space).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)

        self._character.__setattr__(self, 'update_parameters', self.update_parameters)

    def act(self, state, action_space, eval=False):
        # print(hasattr(self._character, 'update_parameters'))
        # TODO discrete actions
        obs = self._translate_obs(state)
        state = torch.FloatTensor(obs).to(self.device).unsqueeze(0)
        if not eval:
            action, _, _ = self.policy.sample(state)
        else:
            with torch.no_grad():
                _, _, action = self.policy.sample(state)
        return action.detach().cpu().numpy()[0]

    def sample_action(self, state):
        """ eval phase need with torch.no_grad() before call this func.

        during training, using sample action to keep exploration, while at evaluation stage,
        using max_prob_action to exploitation.
        :param state:
        :return:
        """
        # TODO same as policy.sample, where to put? which is better?
        action_prob = self.policy(self._translate_obs(state))
        max_prob_action = torch.argmax(action_prob).unsqueeze(0)  # TODO shape?
        assert action_prob.size(1) == self.action_space.n, "Actor output the wrong size"
        action_dist = torch.distributions.Categorical(action_prob)
        action = action_dist.sample().cpu()

        # deal with 0. prob situation
        z = action_prob == 0.0  # can do this op to a np.array
        z = z.float() * 1e-8
        # add a small num(1e-8) to 0. to avoid log 0
        log_action_prob = torch.log(action_prob + z)

        # sampled action, (prob, log_prob), max probability action
        return action, (action_prob, log_action_prob), max_prob_action

    def _translate_obs(self, o):
        obs_width = 11

        board = o['board'].copy()
        agents = np.column_stack(np.where(board > 10))

        for i, agent in enumerate(agents):
            agent_id = board[agent[0], agent[1]]
            if agent_id not in o['alive']:  # < this fixes a bug >
                board[agent[0], agent[1]] = 0
            else:
                board[agent[0], agent[1]] = 11

        obs_radius = obs_width // 2
        pos = np.asarray(o['position'])

        # board
        board_pad = np.pad(board, (obs_radius, obs_radius), 'constant', constant_values=1)
        self.board_cent = board_cent = board_pad[pos[0]:pos[0] + 2 * obs_radius + 1, pos[1]:pos[1] + 2 * obs_radius + 1]

        # bomb blast strength
        bbs = o['bomb_blast_strength']
        bbs_pad = np.pad(bbs, (obs_radius, obs_radius), 'constant', constant_values=0)
        self.bbs_cent = bbs_cent = bbs_pad[pos[0]:pos[0] + 2 * obs_radius + 1, pos[1]:pos[1] + 2 * obs_radius + 1]

        # bomb life
        bl = o['bomb_life']
        bl_pad = np.pad(bl, (obs_radius, obs_radius), 'constant', constant_values=0)
        self.bl_cent = bl_cent = bl_pad[pos[0]:pos[0] + 2 * obs_radius + 1, pos[1]:pos[1] + 2 * obs_radius + 1]

        return np.concatenate((
            board_cent, bbs_cent, bl_cent,
            o['blast_strength'], o['can_kick'], o['ammo']), axis=None)

    def update_parameters(self, memory, batch_size, updates):
        # Sample a batch
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = memory.sample(batch_size=batch_size)

        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)

        with torch.no_grad():
            _, (pi_, log_pi_), _ = self.policy.sample(next_state_batch)

            qf1_next_target, qf2_next_target = self.critic_target(next_state_batch)
            min_qf_next_target = pi_ * (torch.min(qf1_next_target, qf2_next_target) - self.alpha * log_pi_)
            # TODO ? \mean_{a'} Q(s', a') Expected Q?
            min_qf_next_target = min_qf_next_target.mean(dim=1).unsqueeze(-1)
            next_q_value = reward_batch + (1 - mask_batch) * self.gamma * min_qf_next_target
        ###############################################################################
        # Critic losses

        # Two Q-functions to mitigate positive bias in the policy improvement step
        qf1, qf2 = self.critic(state_batch)
        # (256, 6),  (256)
        # print(qf1.shape, action_batch.shape)
        # TODO: UserWarning
        #  Using a target size (torch.Size([256, 1])) that is different to the input size (torch.Size([256, 6])). This will likely lead to incorrect results due to broadcasting.
        #  这里需要test一下正确性
        qf1 = qf1.gather(1, action_batch.view(-1, 1).long())   # 这里的action_batch 是 index
        qf2 = qf2.gather(1, action_batch.view(-1, 1).long())

        qf1_loss = F.mse_loss(qf1, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf2_loss = F.mse_loss(qf2, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]

        ###############################################################################
        # Actor(Policy) losses
        action, (pi, log_pi), _ = self.policy.sample(state_batch)
        # pi, log_pi, _ = self.policy.sample(state_batch)

        qf1_pi, qf2_pi = self.critic(state_batch)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        # Jπ = 𝔼st∼D,εt∼N[α * log π(f(εt;st)|st) − Q(st,f(εt;st))]
        # TODO, discrete action
        #   Jπ = 𝔼st∼D,εt∼N[π * (α * log π(f(εt;st)|st) − Q(st,f(εt;st)))]
        policy_loss = (((self.alpha * log_pi) - min_qf_pi) * pi).mean()
        log_pi = torch.sum(log_pi * pi, dim=1)  # used to calculate alpha loss, still don't understand

        ##############################################################################
        # optimize step
        self.critic_optim.zero_grad()
        qf1_loss.backward()
        self.critic_optim.step()

        self.critic_optim.zero_grad()
        qf2_loss.backward()
        self.critic_optim.step()
        
        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()
            alpha_tlogs = self.alpha.clone()  # For TensorboardX logs
        else:
            alpha_loss = torch.tensor(0.).to(self.device)
            alpha_tlogs = torch.tensor(self.alpha)  # For TensorboardX logs

        if updates % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)

        return qf1_loss.item(), qf2_loss.item(), policy_loss.item(), alpha_loss.item(), alpha_tlogs.item()

    # Save model parameters
    def save_model(self, env_name, suffix="", actor_path=None, critic_path=None):
        if not os.path.exists('models/'):
            os.makedirs('models/')

        if actor_path is None:
            actor_path = "models/sac_actor_{}_{}".format(env_name, suffix)
        if critic_path is None:
            critic_path = "models/sac_critic_{}_{}".format(env_name, suffix)
        print('Saving models to {} and {}'.format(actor_path, critic_path))
        torch.save(self.policy.state_dict(), actor_path)
        torch.save(self.critic.state_dict(), critic_path)
    
    # Load model parameters
    def load_model(self, actor_path, critic_path):
        print('Loading models from {} and {}'.format(actor_path, critic_path))
        if actor_path is not None:
            self.policy.load_state_dict(torch.load(actor_path))
        if critic_path is not None:
            self.critic.load_state_dict(torch.load(critic_path))

