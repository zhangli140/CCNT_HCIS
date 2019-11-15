import gym
import datetime
# from dqn import DQN
# from tmp.core.algo import SacDiscrete
# from tmp.core.config import get_configs
from tensorboardX import SummaryWriter
from ppo.ppo import PPO, Transition
import argparse
from sacd.replay_memory import ReplayMemory
import pandas as pd
import matplotlib.pyplot as plt
import pickle

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
parser.add_argument('--automatic_entropy_tuning', type=bool, default=False, metavar='G',
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
parser.add_argument('--draw', type=bool, default=False,
                    help='run draw (default: False)')
args = parser.parse_args()

def run_toy_env():
    env = gym.make('CartPole-v0')
    env = env.unwrapped
    env.seed(1)
    assert isinstance(env.action_space, gym.spaces.discrete.Discrete)  # we need a discrete-action problem to test
    # agent = DQN(env.action_space.n, 4, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9,
    #             replace_target_iter=500, memory_size=8000)
    # agent = SAC(4, env.action_space, args)
    # agent = SacDiscrete(env.observation_space, env.action_space, configs=get_configs())
    agent = PPO(4, env.action_space.n)
    memo = ReplayMemory(capacity=5000)
    # TesnorboardX
    utc = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    writer = SummaryWriter(logdir=f'runs/{utc}_{agent.name}_{args.env_name}_{args.policy}')

    total_step = 0
    reward_list = []
    loss_list = []
    updates = 0
    alpha = 100
    for i in range(500):
        cum_reward = 0
        obs, done = env.reset(), False
        episode_step = 0
        r_cnt = 0
        while not done:
            # env.render()
            a, action_prob, _action = agent.sample_action(obs)

            action = a.item()
            obs_, r, done, _ = env.step(action)
            r_cnt += r
            if r_cnt >= 500:
                print('early stop!!!')
                done = True
            ############################################################################################
            # 这一段来自莫烦
            # https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow/tree/master/contents/6_OpenAI_gym
            x, x_dot, theta, theta_dot = obs_
            # x 是车的水平位移, 所以 r1 是车越偏离中心, 分越少
            # theta 是棒子离垂直的角度, 角度越大, 越不垂直. 所以 r2 是棒越垂直, 分越高
            r1 = (env.x_threshold - abs(x)) / env.x_threshold  # - 0.8
            r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians  # - 0.5
            reward = r1 + r2  # 总 reward 是 r1 和 r2 的结合, 既考虑位置, 也考虑角度, 这样学习更有效率
            ############################################################################################
            trans = Transition(obs, action, action_prob, reward, obs_)
            agent.store_transition(trans)
            memo.push(obs, action, reward, obs_, done)
            total_step += 1
            episode_step += 1
            if done and len(agent.buffer) >= 32:  # and total_step > 100:
                agent.learn(memo, batch_size=32, updates=i, writer=writer)
                updates += 1
            cum_reward += reward

            obs = obs_
        print(f'Epi:{i+1}, total steps: {total_step}, reward: {cum_reward:.3f}, episode_step:{r_cnt}, alpha:{alpha}')
        reward_list.append(cum_reward)

    data = pd.DataFrame(loss_list)
    print(data.shape)
    with open('./nic.pkl', 'wb') as f:
        pickle.dump([reward_list, data], f)


def draw():
    with open('./nic.pkl', 'rb') as f:
        reward_list, data = pickle.load(f)
    plt.figure(figsize=(16, 12))
    lo, hi = 0, 10
    t = sum(reward_list[:hi])
    rs = [t/10]
    while hi < len(reward_list):
        t += reward_list[hi] - reward_list[hi - 10]
        rs.append(t/10)
        hi += 1

    fig, axes = plt.subplots(2, 3)
    axes[0, 0].plot(list(range(len(rs))), rs)

    assert data.shape[1] == 5, "wrong shape"
    titles = ['critic_1_loss', 'critic_2_loss', 'policy_loss', 'ent_loss', 'alpha']
    for (x, y), i, title in zip([[0, 1], [0, 2], [1, 0], [1, 1], [1, 2]], list(range(5)), titles):
        axes[x, y].plot(list(range(data.shape[0])), data.iloc[:, i])
        axes[x, y].set_title(title)
    fig.savefig('./new.png')


if __name__ == '__main__':
    if args.draw:
        draw()
    else:
        run_toy_env()
    # draw()
