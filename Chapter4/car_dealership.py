import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class CarDealership:
    def __init__(self):
        self.value = np.zeros((21, 21))
        self.old_val = np.zeros((21, 21))
        self.policy = np.zeros((21, 21))
        self.action_space = [-5, 5]
        self.rental_means = [3, 4]
        self.return_means = [3, 2]
        self.discount = 0.9

    def rental_request(self):
        return [np.random.poisson(self.rental_means[0]), np.random.poisson(self.rental_means[1])]

    def rental_return(self):
        return [np.random.poisson(self.return_means[0]), np.random.poisson(self.return_means[1])]

    def step(self, s, a):
        a = max(min(a, s[0]), -s[1])
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
        s = np.clip(s, 0, 20)
        return s, r

    def eval(self):
        self.old_val = self.value
        delta = 0
        for i in range(21):
            for j in range(21):
                target = 0.
                for si in range(20):
                    sp, r = self.step([i, j], self.policy[i, j])
                    target += r + self.discount * self.old_val[int(sp[0]), int(sp[1])]
                target /= 20.
                delta = max(delta, abs(self.value[i, j] - target))
                self.value[i, j] = target
        return delta

    def policy_improvement(self):
        stable = True
        for i in range(21):
            for j in range(21):
                val = []
                for a in range(self.action_space[0], self.action_space[1] + 1):
                    target = 0.
                    for si in range(20):
                        sp, r = self.step([i, j], self.policy[i, j])
                        target += r + self.discount * self.old_val[int(sp[0]), int(sp[1])]
                    target /= 20.
                    val.append(target)
                new_pol = np.argmax(val) + self.action_space[0]
                if new_pol != self.policy[i, j]:
                    stable = False
                self.policy[i, j] = new_pol
        return stable

    def policy_iteration(self):
        stable = False
        epsilon = 0.1
        for p in range(5):
            for i in range(10):
                delta = self.eval()
                if delta < epsilon:
                    break
            print('pol eval')
            stable = self.policy_improvement()
            # self.plot_policy()
            print('pol improv')
            if stable:
                break

    def plot_policy(self):
        plt.figure()
        x = []
        y = []
        v = []
        for i in range(21):
            for j in range(21):
                x.append(i)
                y.append(j)
                v.append(self.policy[i, j])
        plt.scatter(x, y, c=v, cmap='cool')
        plt.show()


def main():
    cd = CarDealership()
    cd.policy_iteration()
    cd.plot_policy()

if __name__ == '__main__':
    main()
