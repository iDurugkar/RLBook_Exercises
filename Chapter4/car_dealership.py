import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import factorial
# import seaborn as sns

def poisson_prob(n, lam):
    prob = np.power(lam, n) / factorial(n) * np.exp(-lam)
    return prob

class CarDealership:
    def __init__(self):
        self.value = np.zeros((11, 11))
        self.old_val = np.zeros((11, 11))
        self.policy = np.zeros((11, 11))
        self.action_space = [-5, 5]
        self.rental_means = [3, 4]
        self.return_means = [3, 2]
        self.discount = 0.9

    def rental_request(self):
        return [np.random.poisson(self.rental_means[0]), np.random.poisson(self.rental_means[1])]

    def rental_return(self):
        return [np.random.poisson(self.return_means[0]), np.random.poisson(self.return_means[1])]

    def step(self, s, a):
        s[0] -= a
        s[1] += a
        r = -2*abs(a)
        rents = self.rental_request()
        rents[0] = min(rents[0], s[0])
        rents[1] = min(rents[1], s[1])
        s[0] -= rents[0]
        s[1] -= rents[1]
        r += 10 * np.sum(rents)
        returns = self.rental_return()
        s[0] += returns[0]
        s[1] += returns[1]
        s = np.clip(s, 0, 10)
        return s, r

    def step_eval(self, s, a):
        s[0] -= int(a)
        s[1] += int(a)
        r = -2 * abs(a)
        if a > 0:
            r += 2
        if s[0] > 5:
            r -= 4
        if s[1] > 5:
            r -= 4
        sp = np.zeros((2, 11,))
        rental_probs = 0
        for i in range(s[0]):
            sp[0, i] = poisson_prob(i, self.rental_means[0])
        sp[0, s[0]] = 1. - np.sum(sp[0, :])
        for i in range(s[1]):
            sp[1, i] = poisson_prob(i, self.rental_means[1])
        try:
            sp[1, s[1]] = 1. - np.sum(sp[1, :])
        except IndexError:
            print('Dafaq')
        for i in range(11):
            rental_probs += 10 * i * sp[1, i] + 10 * i * sp[0, i]
        r += rental_probs
        sp2 = np.zeros((2, 11))
        for i in range(11):
            sp2[0, i] = poisson_prob(i, self.return_means[0])
            sp2[1, i] = poisson_prob(i, self.return_means[1])
        final_sp = np.zeros((2, 11))
        for i in range(s[0] + 1):
            for j in range(11 - s[0] + i):
                final_sp[0, s[0] - i + j] += sp[0, i]*sp2[0, j]
        for i in range(s[1] + 1):
            for j in range(11 - s[1] + i):
                final_sp[1, s[1] - i + j] += sp[1, i]*sp2[1, j]
        next_val = 0.
        for i in range(11):
            for j in range(11):
                next_val += self.old_val[i, j] * final_sp[0, i] * final_sp[1, j]
        g = r + self.discount * next_val
        return g

    def eval(self):
        self.old_val = self.value
        delta = 0
        for i in range(11):
            for j in range(11):
                self.policy[i, j] = max(min(self.policy[i, j], i), -j)
                target = self.step_eval([i, j], self.policy[i, j])
                self.value[i, j] = target
                # target = 0.
                # for si in range(20):
                #     self.policy[i, j] = max(min(self.policy[i, j], i), -j)
                #     sp, r = self.step([i, j], self.policy[i, j])
                #     target += r + self.discount * self.old_val[int(sp[0]), int(sp[1])]
                # target /= 20.
                delta = max(delta, abs(self.value[i, j] - target))
                # self.value[i, j] = target
        return delta

    def policy_improvement(self):
        stable = True
        for i in range(11):
            for j in range(11):
                val = np.ones((11,))*-20.
                action_space = [0, 0]
                action_space[0] = max(self.action_space[0], max(-j, i - 10))
                action_space[1] = min(self.action_space[1] + 1, min(i+1, 11 - j))
                for a in range(action_space[0], action_space[1]):
                    target = self.step_eval([i, j], a)
                    # for si in range(20):
                    #     sp, r = self.step([i, j], self.policy[i, j])
                    #     target += r + self.discount * self.old_val[int(sp[0]), int(sp[1])]
                    # target /= 20.
                    val[a - self.action_space[0]] = target
                new_pol = np.argmax(val) + self.action_space[0]
                if new_pol != self.policy[i, j]:
                    stable = False
                self.policy[i, j] = new_pol
        return stable

    def policy_iteration(self):
        stable = False
        epsilon = 0.1
        for p in range(5):
            for i in range(1000):
                delta = self.eval()
                if delta < epsilon:
                    break
            print(p, delta)
            print('pol eval')
            stable = self.policy_improvement()
            # self.plot_policy()
            print('pol improv')
            if stable:
                break

    def plot_policy(self):
        fig, ax = plt.subplots()
        cax = ax.imshow(self.policy, cmap='coolwarm', origin='lower')
        cbar = fig.colorbar(cax, ticks=[-5, 0, 5])
        cbar.ax.set_yticklabels(['< -5', '0', '> 5'])
        plt.show()
        #
        # plt.figure()
        # x = []
        # y = []
        # v = []
        # for i in range(21):
        #     for j in range(21):
        #         x.append(i)
        #         y.append(j)
        #         v.append(self.policy[i, j])
        # plt.scatter(x, y, c=v, cmap='cool')
        # plt.show()


def main():
    cd = CarDealership()
    cd.policy_iteration()
    cd.plot_policy()

if __name__ == '__main__':
    main()
