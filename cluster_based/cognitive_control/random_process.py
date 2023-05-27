import numpy as np
from scipy.stats import multivariate_normal


class BayesianRegression:
    def __init__(self, data):
        self.data = data
        self.dist_mean = []
        self.dist_cov = []
        self.innovation = []
        
        _, self.Nseqs, self.Npts = data.shape


    def fit(self, start, end):
        L, L1, L2 = self.cholesky_factor()
        fixed_pts = np.array([start, end]).T
        test = self.data[0]
        print(np.mean(L@test.T[0]))
        print(np.mean(L@test.T[10]))
        print(np.mean(L@test.T[20]))


        for i, j in enumerate(self.data):
            mean_seq = np.mean(j, axis=1)
            var_seq = np.var(j, axis=1)
            var_seq_fixed = var_seq[[0, self.Nseqs-1]]
        
            K = np.diag(1/np.sqrt(var_seq_fixed))
            L_tilde = np.block([[L1, L2], [np.zeros(L2.T.shape), K]])
            L_tilde_inv = np.linalg.inv(L_tilde)

            # new_mean = L_tilde_inv @ np.block([np.zeros(self.Nseqs-2), K@fixed_pts[i]])
            new_mean = L_tilde_inv @ np.block([np.zeros(self.Nseqs-2), K@mean_seq[[0, -1]]])
            new_cov = np.linalg.inv(L_tilde.T @ L_tilde)
            # new_cov = L_tilde.T @ L_tilde
            self.dist_mean.append(new_mean)
            self.dist_cov.append(new_cov)
    

    def cholesky_factor(self):
        L = np.zeros((self.Nseqs-2, self.Nseqs))
        
        for i in range(self.Nseqs-2):
            L[i, i:i+3] = -1, 2, -1

        L = L/2

        L1 = L[:, 1:self.Nseqs-1]
        L2 = np.array([L[:, 1], L[:, -1]]).T

        return L, L1, L2


    def sampling(self):
        sample_x = multivariate_normal(self.dist_mean[0], self.dist_cov[0]).rvs(size=1)
        sample_y = multivariate_normal(self.dist_mean[1], self.dist_cov[1]).rvs(size=1)
        sample_pose = multivariate_normal(self.dist_mean[2], self.dist_cov[2]).rvs(size=1)

        sample = np.array([sample_x, sample_y, sample_pose])

        return sample, self.dist_mean, self.dist_cov