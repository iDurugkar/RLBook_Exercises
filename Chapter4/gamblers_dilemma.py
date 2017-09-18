import numpy as np
import matplotlib.pyplot as plt
from Chapter2.bandits_exercise import argmax


class Gamblers:
    def __init__(self):
        self.values = np.zeros((101,))
        self.old_values = np.zeros((101,))
        self.values[100] = 1.
        self.policy = np.ones((100,))
        self.policy[0] = 0.
        self.old_values[100] = 1.
        self.p = 0.05
        self.discount = 1.

    def step_eval(self, s, a):
        r = 0.
        sp = [s - a, s + a]
        spp = [(1 - self.p), self.p]
        val = 0.
        for i in range(2):
            if sp[i] <= 0.:
                val += self.old_values[0] * spp[i]
            elif sp[i] >= 100.:
                val += self.old_values[100] * spp[i]
            else:
                val += self.old_values[int(sp[i])] * spp[i]
        r += val * self.discount
        return r

    def policy_evaluation(self):
        self.old_values = self.values
        delta = 0.
        for i in range(1, 100):
            target = self.step_eval(i, self.policy[i])
            delta = max(delta, abs(self.values[i] - target))
            self.values[i] = target
        return delta

    def policy_improvement(self):
        stable = True
        self.old_values = self.values
        for i in range(1, 100):
            val = np.ones((100,)) * -1.
            for a in range(1, min(i, 100 - i) + 1):
                val[a] = self.step_eval(i, a)
            act = np.argmax(val)
            self.values[i] = val[act]
            if self.policy[i] != act:
                stable = False
            self.policy[i] = act
        return stable

    def value_iteration(self):
        stable = False
        for p in range(300):
            print(p)
            # delta = self.policy_evaluation()
            stable = self.policy_improvement()
            if stable:
                break
        print('done')
        self.plot()

    def plot(self):
        plt.figure(1)
        plt.title('p(h) = %.2f' % self.p)
        plt.subplot(1, 2, 1)
        plt.plot(self.values)
        plt.title('Values')
        plt.xlabel('state')
        plt.ylabel('probability of winning')
        plt.subplot(1, 2, 2)
        plt.plot(self.policy)
        plt.title('Policy')
        plt.xlabel('state')
        plt.ylabel('bet')
        plt.show()


def main():
    prob = Gamblers()
    prob.value_iteration()

if __name__ == '__main__':
    main()