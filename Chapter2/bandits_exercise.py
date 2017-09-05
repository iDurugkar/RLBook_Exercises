from Chapter2.bandit_testbed import KArmedBandit
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def argmax(arr):
    m = np.max(arr)
    indices = np.nonzero(arr == m)
    return np.random.choice(indices[0])


def epsilon_greedy(num_trials=2000, k=10, epsilon=0.1, timesteps=1000, init=0.):

    rewards = []
    prob_optimal = []
    for ep in range(num_trials):
        env = KArmedBandit(k=10, total_time=timesteps)
        rew = []
        opt = []
        env.initialize_episode()
        estimates = np.ones(shape=(k,)) * init
        acted = np.zeros(shape=(k,))
        for i in range(timesteps):
            if np.random.rand() < epsilon:
                action = np.random.randint(k)
            else:
                action = argmax(estimates)
            rew.append(env.step(action=action))
            if action == env.best_arm:
                opt.append(1.)
            else:
                opt.append(0.)
            acted[action] += 1.
            estimates[action] += 0.1 * (rew[-1] - estimates[action])
        rewards.append(rew)
        prob_optimal.append(opt)
    return np.mean(rewards, axis=0), np.mean(prob_optimal, axis=0)


def UCB(num_trials=2000, k=10, timesteps=1000, c=2.):
    rewards = []
    prob_optimal = []
    for ep in range(num_trials):
        env = KArmedBandit(k=10, total_time=timesteps)
        rew = []
        opt = []
        # env.initialize_episode()
        estimates = np.zeros(shape=(k,))
        acted = np.zeros(shape=(k,))
        for i in range(timesteps):
            if i < k:
                action = i
            else:
                confidence = estimates + c * np.sqrt(np.log(i+1)/acted)
                action = argmax(confidence)
            rew.append(env.step(action=action))
            if action == env.best_arm:
                opt.append(1.)
            else:
                opt.append(0.)
            acted[action] += 1.
            estimates[action] += 0.1 * (rew[-1] - estimates[action])
        rewards.append(rew)
        prob_optimal.append(opt)
    return np.mean(rewards, axis=0), np.mean(prob_optimal, axis=0)


def softmax(num_trials=2000, k=10, timesteps=1000, c=2.):

    rewards = []
    prob_optimal = []
    for ep in range(num_trials):
        env = KArmedBandit(k=10, total_time=timesteps)
        rew = []
        opt = []
        # env.initialize_episode()
        estimates = np.zeros(shape=(k,))
        acted = np.zeros(shape=(k,))
        for i in range(timesteps):
            if i < k:
                action = i
            else:
                confidence = estimates * c * np.sqrt((i + 1) / np.log(acted+1))  # np.log(i + 1) / acted  #
                prob = np.exp(confidence) / np.sum(np.exp(confidence))
                action = np.random.choice(range(k), p=prob)  # argmax(prob)
            rew.append(env.step(action=action))
            if action == env.best_arm:
                opt.append(1.)
            else:
                opt.append(0.)
            acted[action] += 1.
            estimates[action] += 0.1 * (rew[-1] - estimates[action])
        rewards.append(rew)
        prob_optimal.append(opt)
    return np.mean(rewards, axis=0), np.mean(prob_optimal, axis=0)


def main():
    eps = [0., 0.01, 0.1]
    for e in eps:
        r, p = epsilon_greedy(epsilon=e)
        plt.figure(1)
        plt.plot(r)
        plt.figure(2)
        plt.plot(p)
    plt.figure(1)
    plt.ylim([0., 1.5])
    plt.legend(['Greedy', r'\epsilon = 0.01', r'\epsilon = 0.1'])
    plt.title(r'Reward comparison of \epsilon-Greedy')
    plt.ylabel('Average Reward')
    plt.xlabel('Steps')
    plt.figure(2)
    plt.ylim([0., 1.])
    plt.legend(['Greedy', r'\epsilon = 0.01', r'\epsilon = 0.1'])
    plt.title(r'Optimality of \epsilon-Greedy')
    plt.ylabel('% Optimal action')
    plt.xlabel('Steps')
    plt.show()


def ucb_run():
    c = 1./16
    avgr = []
    avg_u = []
    cval = []
    for i in range(8):
        r, p = softmax(c=c)
        avgr.append(np.mean(r))
        cval.append(c)
        r, p = UCB(c=c)
        avg_u.append(np.mean(r))
        print('With c=%f, softmax avg = %f and UCB avg = %f' % (c, avgr[-1], avg_u[-1]))
        c *= 2
    plt.semilogx(cval, avgr)
    plt.semilogx(cval, avg_u)
    # r, p = softmax()
    # plt.plot(p)
    # print(np.mean(r))
    # r, p = UCB()
    # plt.plot(r)
    # r, p = epsilon_greedy(epsilon=0.1, init=0.)
    # plt.plot(r)
    # plt.ylim([0., 2.])
    plt.legend(['softmax', 'UCB'])  # , r'\epsilon = 0.1'])
    plt.title(r'Average reward of UCB')
    plt.ylabel('Average Reward')
    plt.xlabel('Steps')
    plt.savefig('UCB_softmax_comp.png')

if __name__ == '__main__':
    ucb_run()
