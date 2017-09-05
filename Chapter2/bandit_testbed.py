import numpy as np


class KArmedBandit:
    def __init__(self, k=10, total_time=1000):
        self.k = k
        self.means = np.random.normal(loc=0., scale=1., size=(self.k,))
        self.best_arm = np.argmax(self.means)
        self.total_time_steps = total_time
        self.time_step = 0

    def plot_arms(self):
        import matplotlib.pyplot as plt
        import seaborn as sns
        plt.figure(1)
        plt.title('Distribution of rewards')
        for i in range(self.k):
            samples = np.random.normal(loc=self.means[i], scale=1., size=(1000,))
            sample_loc = np.random.normal(loc=i+1, scale=0.1, size=(1000,))
            plt.scatter(sample_loc, samples, c='c', alpha=0.5)
            if i == self.best_arm:
                plt.scatter([i+1], [self.means[i]], c='r')
            else:
                plt.scatter([i + 1], [self.means[i]], c='g')
        plt.ylabel('Sampled reward')
        plt.xlabel('Arm')
        plt.show()

    def initialize_episode(self):
        self.time_step = 0

    def step(self, action):
        self.time_step += 1
        if self.time_step > self.total_time_steps:
            print('Episode has ended')
            return None
        return np.random.normal(loc=self.means[action], scale=1.)


def main():
    bandits = KArmedBandit()
    bandits.plot_arms()

if __name__ == '__main__':
    main()