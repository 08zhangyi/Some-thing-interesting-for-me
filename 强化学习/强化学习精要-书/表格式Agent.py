import numpy as np


class TableAgent(object):
    def __init__(self, env):
        self.s_len = env.observation_space.n
        self.a_len = env.action_space.n

        self.r = [env.reward(s) for s in range(0, self.s_len)]
        self.pi = np.array([0 for s in range(0, self.s_len)])
        self.p = np.zeros([self.a_len, self.s_len, self.s_len], dtype=np.float)

        ladder_move = np.vectorize(lambda x: env.ladders[x] if x in env.ladders else x)

        for i, dice in enumerate(env.dices):
            prob = 1.0 / dice
            for src in range(1, 100):
                step = np.arange(dice)
                step += src
                step = np.piecewise(step, [step>100, step<=100], [lambda x: 200-x, lambda x: x])
                step = ladder_move(step)
                for dst in step:
                    self.p[i, src, dst] += prob
        self.p[:, 100, 100] = 1
        self.value_pi = np.zeros((self.s_len))
        self.value_q = np.zeros((self.s_len, self.a_len))
        self.gamma = 0.8

    def play(self, state):
        return self.pi[state]