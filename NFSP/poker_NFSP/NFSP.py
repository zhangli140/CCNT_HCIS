import numpy as np
import tensorflow as tf
import tensorflow.contrib as tfc
from XFP import XFP, LeducRLEnv
from functools import reduce  ###
import threading
import socket
import matplotlib.pyplot as plt



def getState(data, turn):
    state = np.zeros([4, 6, 2], np.int32)  ###

    list1 = data.split('.')
    list1 = list1[1:]
    for i in range(len(list1)):
        list2 = (list1[i]).split(',')
        list2 = list2[1:]
        if len(list2) == 0:
            continue
        for j in range(len(list2)):
            # print(list2)
            state[i, j, 0] = int(list2[j][0])  # action  1:c 2:r 3:f 将fold设为3，不能设为0
            player = -1
            if turn == 0:
                player = 1 if list2[j][1] == '1' else 0
            else:
                player = 0 if list2[j][1] == '1' else 1  ##
            state[i, j, 1] = player  # player  1 2

    return state


def getCard(data):
    t = data[data.rfind(':') + 1:]
    c = ''
    for i in t:
        if i == '|' or i == '/':
            continue
        c += i

    # rank_str = '23456789TJQKA'
    # suit_str = 'shdc'

    rank_dict = {'2': 0, '3': 1, '4': 2, '5': 3, '6': 4, '7': 5, '8': 6, '9': 7, 'T': 8, 'J': 9, 'Q': 10, 'K': 11,
                 'A': 12}
    suit_dict = {'s': 0, 'h': 1, 'd': 2, 'c': 3}

    card = np.zeros([7], np.int32)
    for i in range(0, len(c), 2):
        rank = rank_dict[c[i]]
        suit = suit_dict[c[i + 1]]
        card[int(i / 2)] = suit * 13 + rank + 1

    return card


xfp = XFP(card_num=7, seed=33)
exploits = []
its = []


def evaluate_exploitabality(p1, p2):
    xfp.opponent_policy_p2 = p2
    xfp.compute_p1_best_response()
    br1_op = XFP.convert_q_s_a2greedy_policy(xfp.q_value1_final)
    xfp.opponent_policy_p1 = p1
    xfp.compute_p2_best_response()
    br2_op = XFP.convert_q_s_a2greedy_policy(xfp.q_value2_final)

    # compute optimal best response payoff
    realization_br1_op = XFP.compute_realization(br1_op, p2)
    realization_br2_op = XFP.compute_realization(p1, br2_op)
    e = [XFP.compute_payoff_given_realization(realization_br1_op)[0],
         XFP.compute_payoff_given_realization(realization_br2_op)[1]]
    exploitability = (e[0] + e[1]) / 2.0
    xfp.finish()
    return exploitability, e


class myThread(threading.Thread):

    def __init__(self, id, name, port, nfsp):

        threading.Thread.__init__(self)
        self.id = id
        self.name = name
        self.port = port
        self.nfsp = nfsp

    def run(self):

        clientSocket = socket.socket()
        clientSocket.connect(('192.168.25.128', self.port))  # 要填写服务器ip

        cnt = 0
        x_axis = []
        y_axis = [[],[],[],[]]
        out_log = False
        score_log = 0
        while True:

            # print('准备接受服务器消息')
            recvData = ''
            while len(recvData) == 0:
                # print("666")
                recvData = clientSocket.recv(4096).decode('utf-8')
            # recvData = recvData.strip('\r\n')

            if recvData[-1] == '%':
                # print(self.name + '收到结果消息:%s' % (recvData[:-1]))              #1
                cnt += 1
                score = recvData[recvData.rfind(':') + 1:-1]
                if out_log:
                    print('=====第%d局比赛%s得分:%s=====\n' % (cnt, self.name, score))     #1
                    score_log += int(score[:score.find('.')])

                # 9月27日改动
                clientSocket.send(recvData.encode())
                # print(self.name + '发送结果消息（多余）:%s\n' % (recvData[:-1]))
                # 9月27日改动

                self.nfsp.rl_replay[self.id - 1].add_terminal(score[:score.find('.')])  ## 不要小数
                self.nfsp.epsilon *= 0.99

                if cnt % 1000 == 0:
                    print(cnt)
                    print('test_score:' + str(score_log))
                    for i in range(len(x_axis)):
                        # print(self.name + ":" + str(x_axis[i]) + " " + str(y_axis[i]))
                        f0 = open(self.name + '/sl_loss.txt', 'a')
                        f0.write(self.name + ':' + str(x_axis[i]) + ' ' + str(y_axis[0][i]) + '\n')
                        f0.close()
                        f1 = open(self.name + '/apply_gradients_sl.txt', 'a')
                        f1.write(self.name + ':' + str(x_axis[i]) + ' ' + str(y_axis[1][i]) + '\n')
                        f1.close()
                        f2 = open(self.name + '/rl_loss.txt', 'a')
                        f2.write(self.name + ':' + str(x_axis[i]) + ' ' + str(y_axis[2][i]) + '\n')
                        f2.close()
                        f3 = open(self.name + '/apply_gradients_rl.txt', 'a')
                        f3.write(self.name + ':' + str(x_axis[i]) + ' ' + str(y_axis[3][i]) + '\n')
                        f3.close()


                    x_axis = []
                    y_axis = [[],[],[],[]]
                    # plt.clf()
                    # plt.plot(x_axis, y_axis)
                    # fig = 'log' + str(self.id) + '/' + str(cnt) + '.jpeg'
                    # plt.savefig(fig)

                if cnt == 20000000 - 2 and self.name == 'player1':

                    self.nfsp.m_saver.save(self.nfsp.sess, "Model/nfsp1")

                continue  ## 只调用一次
            else:
                # print(self.name + '收到消息:%s' % (recvData[:recvData.rfind(':')]))         #1
                # print(self.name + '调试:%s' % (recvData))
                # print('当前状态信息:%s' % (recvData[recvData.rfind(':') + 4:]))

                turn = 0 if cnt % 2 == 0 else 1
                myState = getState(recvData[recvData.rfind(':') + 4:], turn)
                myCard = getCard(recvData[:recvData.rfind(':')])
                myTurn = self.id - 1  ##

                action, tag = self.nfsp.choose_action(myTurn, myState, myCard)

                if tag == 'br':  # greedy best response
                    self.nfsp.sl_replay[myTurn].add(myState, myCard, action)
                self.nfsp.rl_replay[myTurn].add(myState, myCard, action, 0, False)
                self.nfsp.iter[myTurn] += 1
                if self.nfsp.iter[myTurn] % self.nfsp.flags.train_frequency == 0 and self.nfsp.iter[
                    myTurn] > self.nfsp.flags.train_start:
                    sl_loss, apply_gradients_sl, rl_loss, apply_gradients_rl = self.nfsp.train(myTurn)
                    x_axis.append(cnt)
                    y_axis[0].append(sl_loss)
                    y_axis[1].append(apply_gradients_sl)
                    y_axis[2].append(rl_loss)
                    y_axis[3].append(apply_gradients_rl)

                # ob = self.env.act(action)

            # recvData += ":c\r\n"
            # fcr (limit)

            # 手动
            recvData = recvData[:recvData.rfind(':')]

            recvData += ':'
            if self.name == "player1" and (cnt < -1):
                str0 = input("请" + self.name + "输入")
                recvData += str0
            elif self.name == "player1" and (cnt >= 1000) and ((cnt % 1000) < 10):
                out_log = True
                str0 = np.random.randint(0, 3)
                if str0 == 0:
                    recvData += 'c'
                if str0 == 1:
                    recvData += 'r'
                if str0 == 2:
                    recvData += 'f'
            else:
                out_log = False
                # recvData += 'c' if action == 0 else 'r'  ####
                if action == 0:
                    recvData += 'c'
                if action == 1:
                    recvData += 'r'
                if action == 2:
                    recvData += 'f'


            recvData += '\r\n'
            # 手动

            clientSocket.send(recvData.encode())
            # print(self.name + '发送消息:%s\n' % (recvData[:-2]))          #1

        clientSocket.close()


def getAction(cnt, ix):
    if (cnt & (1 << ix)) == 0:
        return "c"
    else:
        return "r"

class myThread2(threading.Thread):

    def __init__(self, id, name, port, nfsp):

        threading.Thread.__init__(self)
        self.id = id
        self.name = name
        self.port = port
        self.nfsp = nfsp

    def run(self):

        clientSocket = socket.socket()
        clientSocket.connect(('192.168.25.128', self.port))  # 要填写服务器ip

        cnt = 0
        ix = 0
        while True:

            # print('准备接受服务器消息')
            recvData = ''
            while len(recvData) == 0:
                # print("666")
                recvData = clientSocket.recv(4096).decode('utf-8')
            # recvData = recvData.strip('\r\n')

            if recvData[-1] == '%':
                print(self.name + '收到结果消息:%s' % (recvData[:-1]))        ##
                cnt += 1
                ix = 0
                score = recvData[recvData.rfind(':') + 1:-1]
                # print('=====第%d局比赛%s得分:%s=====\n' % (cnt, self.name, score))

                # self.nfsp.rl_replay[self.id - 1].add_terminal(score[:score.find('.')])  ## 不要小数
                # self.nfsp.epsilon *= 0.99

                if cnt % 1000 == 0:
                    print(cnt)
                    # policy = self.nfsp.compute_self_policy()
                    # exploit, e = evaluate_exploitabality(policy[0], policy[1])       ##
                    # its.append(cnt)
                    # exploits.append(exploit)
                    # print("exploit", exploit)
                    #
                    # plt.clf()
                    # plt.plot(its, exploits)
                    # fig = 'log/1' + str(self.nfsp.flags.seed) + '.jpeg'
                    # plt.savefig(fig)

                continue  ## 只调用一次


            # recvData += ":c\r\n"
            # fcr (limit)

            # 手动
            recvData = recvData[:recvData.rfind(':')]
            # str0 = input("请" + self.name + "输入")
            recvData += ':'
            recvData += getAction(cnt, ix)
            ix += 1
            recvData += '\r\n'
            # 手动

            clientSocket.send(recvData.encode())
            # print(self.name + '发送消息:%s\n' % (recvData[:-2]))

        clientSocket.close()

class NFSP(object):
    def __init__(self, flags):
        self.flags = flags
        self.env = LeducRLEnv(card_num=flags.card_num, seed=flags.seed)
        np.random.seed(flags.seed)
        self.iter = [0, 0]
        self.epsilon = self.flags.epsilon
        self.sl_replay = [ReservoirReplay(flags, self.env), ReservoirReplay(flags, self.env)]
        self.rl_replay = [CircularReplay(flags, self.env), CircularReplay(flags, self.env)]

        tf_config = tf.ConfigProto(
            allow_soft_placement=True
        )
        tf_config.gpu_options.allow_growth = True
        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph, config=tf_config)
        self.sess.__enter__()
        tf.set_random_seed(flags.seed)

        self.ops = [{}, {}]
        for i in range(2):
            with tf.variable_scope("player" + str(i)):
                self.ops[i]['state_history_ph'] = tf.placeholder(tf.int8, [None] + self.env.state_history_space,
                                                                 "state_history")
                self.ops[i]['state_card_ph'] = tf.placeholder(tf.int8, [None] + self.env.state_card_space,
                                                              "state_card")
                self.ops[i]['state_history_ph2'] = tf.placeholder(tf.int8, [None] + self.env.state_history_space,
                                                                  "state_history2")
                self.ops[i]['state_card_ph2'] = tf.placeholder(tf.int8, [None] + self.env.state_card_space,
                                                               "state_card2")

                with tf.variable_scope('current_q'):
                    self.ops[i]['q_logits_s'] = self._build_inference(self.ops[i]['state_history_ph'],
                                                                      self.ops[i]['state_card_ph'])

                with tf.variable_scope('old_q'):
                    self.ops[i]['q_logits_s2_old'] = self._build_inference(self.ops[i]['state_history_ph2'],
                                                                           self.ops[i]['state_card_ph2'])

                with tf.variable_scope('average_pi'):
                    self.ops[i]['pi_logits_s'] = self._build_inference(self.ops[i]['state_history_ph'],
                                                                       self.ops[i]['state_card_ph'], softmax=True)

                assign_ops = []
                for (cur, old) in zip(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                                        scope=tf.get_variable_scope().name + '.*current'),
                                      tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                                        scope=tf.get_variable_scope().name + '.*old')):
                    assign_ops.append(tf.assign(old, cur))
                self.ops[i]['copy'] = tf.group(*assign_ops, name="copy")

                self.ops[i]['global_step'] = tf.get_variable("global_step", [], tf.int64,
                                                             tf.constant_initializer(0), trainable=False)
                self.ops[i]['action_ph'] = tf.placeholder(tf.int8, [None])
                self.ops[i]['reward_ph'] = tf.placeholder(tf.int8, [None])
                self.ops[i]['terminal_ph'] = tf.placeholder(tf.int8, [None])  # 1.0 is terminal
                self.ops[i]['apply_gradients_sl'], self.ops[i]['apply_gradients_rl'], self.ops[i]['sl_loss'], \
                self.ops[i]['rl_loss'] = \
                    self._build_train(self.ops[i]['action_ph'],
                                      self.ops[i]['pi_logits_s'],
                                      self.ops[i]['reward_ph'],
                                      self.ops[i]['terminal_ph'],
                                      self.ops[i]['q_logits_s'],
                                      self.ops[i]['q_logits_s2_old'],
                                      self.ops[i]['global_step'])

        self.m_saver = tf.train.Saver()
        self.init = tf.global_variables_initializer()
        self.sess.run(self.init)
        self.sess.run([self.ops[0]['copy'], self.ops[1]['copy']])

    def _build_inference(self, state_history_ph, state_card_ph, reuse=False, softmax=False):
        state_history = tf.reshape(tf.cast(state_history_ph, tf.float32),
                                   [-1] + [reduce(lambda x, y: x * y, self.env.state_history_space)])

        state_card = tf.cast(state_card_ph, tf.float32)
        state = tf.concat([state_card, state_history], axis=1)
        logits = self._network(state, reuse, softmax)
        return logits

    def _build_train(self, action, pi_logits_s, reward, terminal, q_logits_s, q_logits_s2_old, global_step):
        action = tf.cast(action, tf.int32)
        one_hot_actions = tf.one_hot(action, self.env.action_space, dtype=tf.float32)
        neglog_pi = tf.nn.softmax_cross_entropy_with_logits(logits=pi_logits_s, labels=one_hot_actions)

        reward = tf.cast(reward, tf.float32)
        terminal = tf.cast(terminal, tf.float32)
        target = reward + (1.0 - terminal) * tf.reduce_max(q_logits_s2_old, axis=1)
        q_s_a = tf.reduce_sum(q_logits_s * one_hot_actions, axis=1)
        loss = tf.reduce_mean(tf.square(q_s_a - tf.stop_gradient(target)))

        optimizer_sl = tf.train.GradientDescentOptimizer(self.flags.lr_sl)  # .minimize(neglog_pi)
        optimizer_rl = tf.train.GradientDescentOptimizer(self.flags.lr_rl)  # .minimize(loss)

        grad_var_list_sl = optimizer_sl.compute_gradients(neglog_pi)
        grad_var_list_rl = optimizer_rl.compute_gradients(loss)

        apply_gradients_sl = optimizer_sl.apply_gradients(grad_var_list_sl, global_step)
        apply_gradients_rl = optimizer_rl.apply_gradients(grad_var_list_rl)

        return apply_gradients_sl, apply_gradients_rl, neglog_pi, loss
        # return optimizer_sl, optimizer_rl, neglog_pi, loss

    def _linear_layer(self, linear_in, dim, hiddens):
        weights = tf.get_variable('weights', [dim, hiddens], tf.float32,
                                  initializer=tfc.layers.variance_scaling_initializer(mode='FAN_AVG', uniform=True))
        bias = tf.get_variable('bias', [hiddens], tf.float32,
                               initializer=tf.constant_initializer(0.1))
        pre_activations = tf.add(tf.matmul(linear_in, weights), bias)
        return pre_activations

    def _network(self, state, reuse=False, softmax=False):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        with tf.variable_scope('linear1'):
            dim = state.get_shape().as_list()[1];
            hiddens = 64
            pre_activations = self._linear_layer(state, dim, hiddens)
            linear1 = tf.nn.relu(pre_activations)
        with tf.variable_scope('linear2'):
            dim = hiddens;
            hiddens = self.env.action_space
            logits = self._linear_layer(linear1, dim, hiddens)
        if softmax:
            logits = tf.nn.softmax(logits)
        return logits

    def choose_action(self, position, state_history, state_card):
        if np.random.rand() < self.flags.anticipatory:  # epsilon greedy
            if np.random.rand() < self.epsilon:  # explore
                action = np.random.randint(0, self.env.action_space)
            else:
                q_logits_s = self.sess.run(self.ops[position]['q_logits_s'],
                                           feed_dict={self.ops[position]['state_history_ph']: [state_history],
                                                      self.ops[position]['state_card_ph']: [state_card]})
                action = np.argmax(q_logits_s[0])
            return action, 'br'  # best response
        else:
            prob = self.sess.run(self.ops[position]['pi_logits_s'],
                                 feed_dict={self.ops[position]['state_history_ph']: [state_history],
                                            self.ops[position]['state_card_ph']: [state_card]})
            # prob = np.exp(pi_logits_s[0])
            # prob = prob / np.sum(prob)
            action = np.random.choice(self.env.action_space, p=prob[0])
            return action, 'avg'  # average

    def play_game(self):

        player1 = myThread(1, "player1", 8000, self)                  #
        player2 = myThread(2, "player2", 8001, self)
        player1.start()
        player2.start()

        # ob = self.env.reset()
        # while True:
        #     position = ob['turn']
        #     if ob['turn'] == -1:
        #         self.rl_replay[0].add_terminal(ob['payoff'][0])
        #         self.rl_replay[1].add_terminal(ob['payoff'][1])
        #         break
        #     state_history = ob['state']
        #     state_card = ob['card']
        #     action, tag = self.choose_action(position, state_history, state_card)
        #     if tag == 'br':  # greedy best response
        #         self.sl_replay[position].add(state_history, state_card, action)
        #     self.rl_replay[position].add(state_history, state_card, action, 0, False)
        #     self.iter[position] += 1
        #     if self.iter[position] % self.flags.train_frequency == 0 and self.iter[position] > self.flags.train_start:
        #         self.train(position)
        #
        #     ob = self.env.act(action)

    def train(self, position):
        batch_state_history_buffer, batch_state_card_buffer, batch_action_buffer = self.sl_replay[
            position].get_random_batch()
        global_step, sl_loss, apply_gradients_sl = self.sess.run([self.ops[position]['global_step'], self.ops[position]['sl_loss'], self.ops[position]['apply_gradients_sl']],
                                       feed_dict={self.ops[position]['state_history_ph']: batch_state_history_buffer,
                                                  self.ops[position]['state_card_ph']: batch_state_card_buffer,
                                                  self.ops[position]['action_ph']: batch_action_buffer})
        batch_state_history_buffer, batch_state_card_buffer, batch_action_buffer, \
        batch_reward_buffer, batch_terminal_buffer, batch_state_history_buffer2, \
        batch_state_card_buffer2 = self.rl_replay[position].get_random_batch()
        ###
        rl_loss, apply_gradients_rl = self.sess.run([self.ops[position]['rl_loss'], self.ops[position]['apply_gradients_rl']],
                          feed_dict={self.ops[position]['state_history_ph']: batch_state_history_buffer,
                                     self.ops[position]['state_card_ph']: batch_state_card_buffer,
                                     self.ops[position]['action_ph']: batch_action_buffer,
                                     self.ops[position]['reward_ph']: batch_reward_buffer,
                                     self.ops[position]['terminal_ph']: batch_terminal_buffer,
                                     self.ops[position]['state_history_ph2']: batch_state_history_buffer2,
                                     self.ops[position]['state_card_ph2']: batch_state_card_buffer2})
        if global_step % self.flags.refit == 0:
            self.sess.run(self.ops[position]['copy'])


        # print('rl_loss\n')
        # print(rl_loss)
        # print('\n')
        # print('apply_gradients_rl\n')
        # print(apply_gradients_rl)
        # print('\n')

        return sl_loss, apply_gradients_sl, rl_loss, apply_gradients_rl

        # print 'train policy {:d} at step {:d}, global_step={:d}, epsilon={:.4f}'.format(position, self.iter[position],global_step, self.epsilon)

    def My_compute_self_policy(self):
        policy = [{}, {}]
        cards = [[],[],[],[]]     ## c52_2(1326),c48_3,c45_1,c44_1
        states = [[],[],[],[]]    ## 4,6,2
                                  ## 4*(0+1+4+4^2+4^3+4^4+4^5) + .. + ... + .... 5360*4096..
        for player in range(2):
            for round in range(4):
                for card in cards[round]:
                    for state in states[round]:
                        q_p, pi = self.sess.run([self.ops[player]['q_logits_s'], self.ops[player]['pi_logits_s']],
                                                feed_dict={self.ops[player]['state_history_ph']: [state],
                                                           self.ops[player]['state_card_ph']: [card]})
                        q_max_a = np.argmax(q_p[0])
                        prob = pi[0]
                        # prob = np.exp(pi[0])
                        # prob = prob / np.sum(prob)
                        prob = prob * (1 - self.flags.anticipatory)
                        prob[q_max_a] += self.flags.anticipatory * 1.0
                        policy[player][card + history] = prob

        return policy


    def compute_self_policy(self):
        policy = [{}, {}]
        for cards in XFP.possible_cards_list:
            for history in LeducRLEnv.history_string2vector:
                state_card = np.zeros([self.env.card_num], np.int32)
                state_history = LeducRLEnv.history_string2vector[history]
                pround = 1 if history in XFP.round1_states_set else 2
                if history in XFP.player1_states_set:
                    state_card[int(cards[0])] = 1
                    card = cards[0]
                    if pround == 2:
                        state_card[int(self.env.card_num / 2) + int(cards[1])] = 1  ##
                        card = cards[:2]
                    if card + history in policy[0]: continue
                    q_p, pi = self.sess.run([self.ops[0]['q_logits_s'], self.ops[0]['pi_logits_s']],
                                            feed_dict={self.ops[0]['state_history_ph']: [state_history],
                                                       self.ops[0]['state_card_ph']: [state_card]})
                    q_max_a = np.argmax(q_p[0])
                    prob = pi[0]
                    # prob = np.exp(pi[0])
                    # prob = prob / np.sum(prob)
                    prob = prob * (1 - self.flags.anticipatory)
                    prob[q_max_a] += self.flags.anticipatory * 1.0
                    policy[0][card + history] = prob
                else:
                    state_card[int(cards[2])] = 1
                    card = cards[2]
                    if pround == 2:
                        state_card[int(self.env.card_num / 2) + int(cards[1])] = 1  ##
                        card = cards[1:]
                    if card + history in policy[1]: continue
                    q_p, pi = self.sess.run([self.ops[1]['q_logits_s'], self.ops[1]['pi_logits_s']],
                                            feed_dict={self.ops[1]['state_history_ph']: [state_history],
                                                       self.ops[1]['state_card_ph']: [state_card]})
                    q_max_a = np.argmax(q_p[0])
                    prob = pi[0]
                    # prob = np.exp(pi[0])
                    # prob = prob / np.sum(prob)
                    prob = prob * (1 - self.flags.anticipatory)
                    prob[q_max_a] += self.flags.anticipatory * 1.0
                    policy[1][card + history] = prob

        return policy


class ReservoirReplay(object):  # sl
    def __init__(self, flags, env):
        self.flags = flags
        self.env = env
        self.state_history_buffer = np.zeros([self.flags.sl_len] + self.env.state_history_space, np.int8)
        self.state_card_buffer = np.zeros([self.flags.sl_len] + self.env.state_card_space, np.int8)
        self.action_buffer = np.zeros([self.flags.sl_len], np.int8)
        self.size = 0
        self.top = 0

    def add(self, state_history, state_card, action):
        if self.size < self.flags.sl_len:
            self.state_history_buffer[self.top] = state_history
            self.state_card_buffer[self.top] = state_card
            self.action_buffer[self.top] = action
            self.top += 1
            self.size += 1
        else:
            prob_add = float(self.flags.sl_len) / float(self.top + 1)
            if np.random.rand() < prob_add:
                index = np.random.randint(0, self.flags.sl_len)
                self.state_history_buffer[index] = state_history
                self.state_card_buffer[index] = state_card
                self.action_buffer[index] = action
            # index = np.random.randint(0, self.flags.sl_len + 1)
            # if index < self.flags.sl_len:
            #     self.state_history_buffer[index] =  state_history
            #     self.state_card_buffer[index] = state_card
            #     self.action_buffer[index] = action

            self.top += 1

    def get_random_batch(self):
        indices = np.random.randint(0, self.size, self.flags.batch)
        batch_state_history_buffer = np.take(self.state_history_buffer, indices, axis=0)
        batch_state_card_buffer = np.take(self.state_card_buffer, indices, axis=0)
        batch_action_buffer = np.take(self.action_buffer, indices)
        return batch_state_history_buffer, batch_state_card_buffer, batch_action_buffer


class CircularReplay(object):  # rl
    def __init__(self, flags, env):
        self.flags = flags
        self.env = env
        self.state_history_buffer = np.zeros([self.flags.rl_len] + self.env.state_history_space, np.int8)
        self.state_card_buffer = np.zeros([self.flags.rl_len] + self.env.state_card_space, np.int8)
        self.action_buffer = np.zeros([self.flags.rl_len], np.int8)
        self.reward_buffer = np.zeros([self.flags.rl_len], np.int8)
        self.terminal_buffer = np.zeros([self.flags.rl_len], np.int8)
        self.size = 0
        self.top = 0
        self.bottom = 0

    def add(self, state_history, state_card, action, reward, terminal):
        self.state_history_buffer[self.top] = state_history
        self.state_card_buffer[self.top] = state_card
        self.action_buffer[self.top] = action
        self.reward_buffer[self.top] = reward
        self.terminal_buffer[self.top] = terminal
        if self.size == self.flags.rl_len:
            self.bottom = (self.bottom + 1) % self.flags.rl_len
        else:
            self.size += 1
        self.top = (self.top + 1) % self.flags.rl_len

    def add_terminal(self, reward):
        last_top = (self.top - 1) % self.flags.rl_len
        self.reward_buffer[last_top] = reward
        self.terminal_buffer[last_top] = True

    def get_random_batch(self):
        indices = np.random.randint(0, self.size, self.flags.batch)
        indices2 = indices + 1
        batch_state_history_buffer = np.take(self.state_history_buffer, indices, axis=0)
        batch_state_card_buffer = np.take(self.state_card_buffer, indices, axis=0)
        batch_state_history_buffer2 = np.take(self.state_history_buffer, indices2, axis=0, mode='wrap')
        batch_state_card_buffer2 = np.take(self.state_card_buffer, indices2, axis=0, mode='wrap')
        batch_action_buffer = np.take(self.action_buffer, indices)
        batch_reward_buffer = np.take(self.reward_buffer, indices)
        batch_terminal_buffer = np.take(self.terminal_buffer, indices)
        return batch_state_history_buffer, batch_state_card_buffer, batch_action_buffer, batch_reward_buffer, \
               batch_terminal_buffer, batch_state_history_buffer2, batch_state_card_buffer2
