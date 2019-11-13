
import numpy as np
from obs_onebyone import obs


def trans_obs(obs):
    o = np.zeros(shape=(16, 8, 8))
    board = obs['board']
    for i in range(12):
        o[i][board == i] = 1.
    o[12] = obs['bomb_blast_strength']
    o[13] = obs['bomb_life']
    o[14] = obs['bomb_moving_direction']
    o[15] = obs['flame_life']

    vec = np.array([obs['ammo'], obs['can_kick'], obs['blast_strength'], obs['step_count']/800], dtype=np.float32)
    return o, vec


def compute_reward(obs, obs_, rewards):
    # compute reward
    reward = 0.
    # todo
    # pick up kick 0.02
    if obs_['can_kick'] and not obs['can_kick']:
        reward += 0.02
    # blast strength 0.01
    reward += (obs_['blast_strength'] - obs['blast_strength']) * 0.01
    # pick up ammo 0.01
    if obs_['ammo'] > obs['ammo']:  # drop a bomb will also cause this.
        reward += 0.01
    # draw 0/0
    return reward + rewards[0]


def translate_observation(obs):
    obs_width = 8

    board = obs['board'].copy()
    agents = np.column_stack(np.where(board > 10))

    for i, agent in enumerate(agents):
        agent_id = board[agent[0], agent[1]]
        if agent_id not in obs['alive']:  # < this fixes a bug >
            board[agent[0], agent[1]] = 0
        else:
            board[agent[0], agent[1]] = 11

    obs_radius = obs_width // 2
    pos = np.asarray(obs['position'])

    # board
    board_pad = np.pad(board, (obs_radius, obs_radius), 'constant', constant_values=1)
    board_cent = board_pad[pos[0]:pos[0] + 2 * obs_radius + 1, pos[1]:pos[1] + 2 * obs_radius + 1]

    # bomb blast strength
    bbs = obs['bomb_blast_strength']
    bbs_pad = np.pad(bbs, (obs_radius, obs_radius), 'constant', constant_values=0)
    bbs_cent = bbs_pad[pos[0]:pos[0] + 2 * obs_radius + 1, pos[1]:pos[1] + 2 * obs_radius + 1]

    # bomb life
    bl = obs['bomb_life']
    bl_pad = np.pad(bl, (obs_radius, obs_radius), 'constant', constant_values=0)
    bl_cent = bl_pad[pos[0]:pos[0] + 2 * obs_radius + 1, pos[1]:pos[1] + 2 * obs_radius + 1]

    return np.concatenate((
        board_cent, bbs_cent, bl_cent,
        obs['blast_strength'], obs['can_kick'], obs['ammo']), axis=None)


def plot_reward():
    import pandas as pd
    import matplotlib.pyplot as plt
    data = pd.read_csv('run_reward.csv')
    moving_mean = data['Value'].rolling(window=20).mean()
    plt.figure()
    plt.plot(np.arange(len(moving_mean)), moving_mean)
    plt.show()


if __name__ == '__main__':
    plot_reward()
