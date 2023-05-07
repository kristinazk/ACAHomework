import numpy as np
from sklearn.datasets import make_blobs, make_moons, make_circles
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


class SpectralClustering_Own:
    def __init__(self, k, laplacian='normalized', construct='KNN'):
        self.k = k
        self.laplacian = laplacian
        self.construct = construct
        self.laplacian_matrix = None

    def calc_affinity(self, X):
        # To calculate affinity we find 5 neighbors and calculate the gaussian similarity
        dist_matr = self.dist_calc(X)
        median = np.median(dist_matr)  # Will be used as sigma
        affinity_matr = np.zeros((len(X), len(X)))
        if self.construct == 'KNN':
            for i in range(len(X)):
                k_neighbors = NearestNeighbors(n_neighbors=5, algorithm='ball_tree').fit(np.delete(X, i, axis=0))
                _, indices = k_neighbors.kneighbors(X[i].reshape(1, -1))

                for ind in indices[0]:
                    affinity_matr[i][ind] = np.exp((-dist_matr[i][ind] ** 2) / median ** 2)

        # In case of full I get an error (the reason is unknown to me)
        if self.construct == 'Full':
            for i in range(len(X)):
                for j in range(len(X)):
                    affinity_matr[i][j] = np.exp((-dist_matr[i][j] ** 2) / median ** 2)

        return (affinity_matr + affinity_matr.T) / 2  # Obtaining a symmetric matrix

    def calc_degree(self, X):
        degree_matrix = np.diag(self.calc_affinity(X).sum(axis=1))
        return degree_matrix

    def fit(self, X):
        degree_matr = self.calc_degree(X)
        affinity_matr = self.calc_affinity(X)
        identity = np.identity(len(X))

        if self.laplacian == 'unnormalized':
            self.laplacian_matrix = degree_matr - affinity_matr

        if self.laplacian == 'normalized':
            self.laplacian_matrix = identity - np.linalg.inv(degree_matr).dot(affinity_matr)

        if self.laplacian == 'symmetric_norm':
            neg_d = np.diag(np.power(np.diag(degree_matr), -0.5))
            self.laplacian_matrix = identity - neg_d.dot(affinity_matr).dot(neg_d)

    def dist_calc(self, X):
        dist_matr = np.zeros((len(X), len(X)))
        for i in range(len(X)):
            for j in range(len(X)):
                dist_matr[i][j] = np.linalg.norm(X[i] - X[j])
        return dist_matr

    def predict(self):
        eigenvals, eigenvects = np.linalg.eig(self.laplacian_matrix)

        eigenvects = eigenvects[:, np.argsort(eigenvals)]

        kmeans = KMeans(n_clusters=self.k, n_init='auto')
        kmeans.fit(eigenvects[:, 1: self.k])

        return kmeans.labels_

    @staticmethod
    def accuracy(y_test, y_pred):
        return sum(y_test == y_pred) / len(y_test)


X_moons, y_moons = make_moons(n_samples=1000, noise=0.08, random_state=0)

scaler = StandardScaler()

sc = SpectralClustering_Own(k=2, construct='KNN', laplacian='normalized')

sc.fit(scaler.fit_transform(X_moons))

print(sc.predict())

print(sc.accuracy(y_moons, sc.predict()))

