import math
import numpy as np

EPS = 1e-8


class MCTS():
    """
    This class handles the MCTS tree.
    """

    def __init__(self, game, nnet):
        self.game = game
        self.nnet = nnet
        self.Qsa = {}  # stores Q values for s,a (as defined in the paper)
        self.Nsa = {}  # stores #times edge s,a was visited
        self.Ns = {}  # stores #times board s was visited
        self.Ps = {}  # stores initial policy (returned by neural net)

        self.numMCTSSims = 25
        self.cpuct = 1
        self.actionSize = 3

    def getStateRepresentation(self, position, history, card):
        a = str(position)
        b = ''.join([str(x) for x in history.flatten().tolist()])
        c = ''.join([str(x) for x in card.flatten().tolist()])
        return a + b + c

    def numToAction(self, n):
        if n == 0:
            return 'c'
        if n == 1:
            return 'r'
        if n == 2:
            return 'f'

    def randomCard(self, card):                  ###
        random_card = np.zeros([2], np.int32)
        random_card[0] = np.random.randint(1, 53)
        while random_card[0] == card[0] or random_card[0] == card[1]:
            random_card[0] = np.random.randint(1, 53)

        random_card[1] = np.random.randint(1, 53)
        while random_card[1] == card[0] or random_card[1] == card[1] or random_card[1] == random_card[0]:
            random_card[1] = np.random.randint(1, 53)

        return random_card

    def getActionProb(self, position, history, card, data, temp=1):
        """
        This function performs numMCTSSims simulations of MCTS starting from
        canonicalBoard.

        Returns:
            probs: a policy vector where the probability of the ith action is
                   proportional to Nsa[(s,a)]**(1./temp)
        """
        data = data[:data.rfind(':')]

        for i in range(self.numMCTSSims):
            random_card = self.randomCard(card)
            print("TEST 8")
            self.search(position, history, card, data, random_card)


        s = self.getStateRepresentation(position, history, card)
        counts = [self.Nsa[(s, a)] if (s, a) in self.Nsa else 0 for a in range(self.actionSize)]

        if temp == 0:
            bestA = np.argmax(counts)
            probs = [0] * len(counts)
            probs[bestA] = 1
            return probs

        counts = [x ** (1. / temp) for x in counts]
        probs = [x / float(sum(counts)) for x in counts]        # c r f  0 1 2
        return probs

    def getAvgActionProb(self, position, history, card):
        """
        this function get action from average net
        Returns:
            probs: a policy vector where the probability of the action got from avg net
        """
        pi, _ = self.nnet.choose_action_avg(position, history, card)  # opponent average policy
        # print("pi",pi)

        sum_Ps_s = np.sum(pi)
        pi /= sum_Ps_s  # renormalize

        pi = pi.tolist()
        return pi

    def search(self, position, history, card, data, random_card):
        """
        This function performs one iteration of MCTS. It is recursively called
        till a leaf node is found. The action chosen at each node is one that
        has the maximum upper confidence bound as in the paper.

        Once a leaf node is found, the neural network is called to return an
        initial policy P and a value v for the state. This value is propogated
        up the search path. In case the leaf node is a terminal state, the
        outcome is propogated up the search path. The values of Ns, Nsa, Qsa are
        updated.

        NOTE: the return values are the negative of the value of the current
        state. This is done since v is in [-1,1] and if v is the value of a
        state for the current player, then its value is -v for the other player.

        Returns:
            v: the negative of the value of the current canonicalBoard
        """

        s = self.getStateRepresentation(position, history, card)

        if s not in self.Ps:       #never visited
            # leaf node
            print("TEST 9")
            self.Ps[s], v = self.nnet.choose_action_avg(position, history, card)   # 0 1 2   c r f

            sum_Ps_s = np.sum(self.Ps[s])
            self.Ps[s] /= sum_Ps_s  # renormalize

            self.Ns[s] = 0
            return v
        print("TEST 10")
        cur_best = -float('inf')
        best_act = -1

        # pick the action with the highest upper confidence bound
        for a in range(self.actionSize):
            if (s, a) in self.Qsa:
                u = self.Qsa[(s, a)] + self.cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s]) / (
                1 + self.Nsa[(s, a)])
            else:
                u = self.cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s] + EPS)  # Q = 0 ?

            if u > cur_best:
                cur_best = u
                best_act = a

        a = best_act
        print("TEST 11")
        # change
        # for next player avg policy
        data += ':' + self.numToAction(a)
        data += '*' + ("%02d" % card[0]) + '*' + ("%02d" % card[1]) + '*' + ("%02d" % random_card[0]) + '*' + ("%02d" % random_card[1])
        next_position, next_history, next_card, next_data = self.game.query(data)

        while 1:
            if next_data[-1] == '%':
                # terminal node
                score = next_data[next_data.rfind(':') + 1:-1]
                score = int(score[:score.find('.')])
                score = 1 if score > 0 else -1
                v = -score if next_position != position else score  # for best player
                print("TEST 15")
                break
            else:
                print("TEST 17")
                if next_position == position:
                    v = self.search(next_position, next_history, next_card, next_data, random_card)
                    break                          ####
                else:
                    # next_card[0] = random_card[0]
                    # next_card[1] = random_card[1]
                    pi = self.getAvgActionProb(next_position, next_history, np.append(random_card, card[2:]))
                    print("np.sum(pi):"+str(np.sum(pi)))
                    print(pi)
                    pi = pi/np.sum(pi)
                    print("np.sum(pi):" + str(np.sum(pi)))
                    print(pi)
                    next_a = np.random.choice(len(pi), p=pi)
                    next_data += ':' + self.numToAction(next_a)
                    next_data += '*' + ("%02d" % card[0]) + '*' + ("%02d" % card[1]) + '*' + ("%02d" % random_card[0]) + '*' + ("%02d" % random_card[1])
                    next_position, next_history, next_card, next_data = self.game.query(next_data)
        #####

        if (s, a) in self.Qsa:
            self.Qsa[(s, a)] = (self.Nsa[(s, a)] * self.Qsa[(s, a)] + v) / (self.Nsa[(s, a)] + 1)
            self.Nsa[(s, a)] += 1

        else:
            self.Qsa[(s, a)] = v
            self.Nsa[(s, a)] = 1

        self.Ns[s] += 1
        print("TEST 16")
        return v
