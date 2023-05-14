import numpy as np
from sklearn.metrics import pairwise_distances


class t_SNE:
    def __init__(self, num_components=2, num_it=100, rate=10, momentum=1, perplexity=30):
        self.num_it = num_it
        self.rate = rate
        self.momentum = momentum
        self.perplexity = perplexity
        self.num_components = num_components

    def fit(self, X):
        sigmas = np.ones(len(X))
        distances_X = pairwise_distances(X)
        y = np.random.rand(len(X), self.num_components)
        distances_y = pairwise_distances(y)

        for i in range(len(X)):
            sigmas[i] = self.set_sigma(self.calc_dist(distances_X[i], sigmas[i]))
            probas = self.calc_dist(distances_X[i], sigmas[i])
            qs = self.calc_qs(distances_y[i])
            prev_val = y[i]
            for _ in range(self.num_it):
                y[i] += self.rate * self.calc_gradient(probas, qs, y, i) + self.momentum * (y[i] - prev_val)
                prev_val = y[i]

        return y

    @staticmethod
    def calc_dist(dists, sigma):
        probas = np.exp(-dists ** 2 / 2 * sigma)

        return probas / (np.sum(probas) - 1)

    @staticmethod
    def perplexity_calc(probas):
        entropy = -np.sum(probas * np.log2(probas))
        return 2 ** entropy

    def set_sigma(self, distances, max_iter=100):
        min_sigma = 0
        max_sigma = np.inf
        sigma = 1

        for _ in range(max_iter):
            perp_own = self.perplexity_calc(distances)

            if np.abs(perp_own - self.perplexity) <= 1e-4:
                return sigma

            if (perp_own - self.perplexity) > 0:
                max_sigma = sigma
                sigma = (sigma + min_sigma) / 2.0
            else:
                min_sigma = sigma
                if max_sigma == np.inf:
                    sigma *= 2.0
                else:
                    sigma = (sigma + max_sigma) / 2.0

        return sigma

    @staticmethod
    def calc_qs(distances):
        probas = (1 + distances) ** -1
        return probas / (np.sum(probas) - 1)

    @staticmethod
    def calc_gradient(prob_p, prob_q, y, idx):
        summary = 0
        for i in range(len(y)):
            summary += (prob_p[i] - prob_q[i]) * (y[idx] - y[i]) * (1 + np.linalg.norm(y[idx] - y[i]) ** 2) ** -1

        return 4 * summary


tsne = t_SNE(num_components=4)

data = np.random.rand(100, 20)

print(tsne.fit(data))
