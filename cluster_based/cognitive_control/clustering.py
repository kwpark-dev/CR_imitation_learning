import numpy as np
from collections import Counter
from scipy.stats import multivariate_normal as mv



def kmeans(cluster_data, num_cluster, dim, eps=0.0001):
    # choose random "num_cluster" centeral points among the data to initialize the k-means 
    # init covariance as 0 because no samples yet
    center_idx = np.random.choice(len(cluster_data), num_cluster, replace=False)
    centroid_center = cluster_data[center_idx]
    centroid_cov = np.zeros((num_cluster, dim, dim))
    centroid_weight = np.zeros(num_cluster) # initial params for EM 
    
    # loop to find convergent (e.g. current-prev < eps) center
    while True:
        prev = centroid_center.copy()
        inliers = [[] for _ in range(num_cluster)]

        for pts in cluster_data:
            dists = np.array([np.linalg.norm(pts-j, ord=2) for j in centroid_center])
            idx = np.argmin(dists)

            inliers[idx].append(pts)

        inliers = np.array(inliers, dtype='object')
        centroid_center = np.array([np.mean(i, axis=0) for i in inliers]) # update center
        convergence = sum([np.linalg.norm(k-l) for k, l in zip(centroid_center, prev)]) 

        if convergence < eps : # check covergence. If it is, calc covariance
            N = len(cluster_data)
    
            for i, j in enumerate(inliers):
                centroid_cov[i] = np.cov(np.array(j).T)
                centroid_weight[i] = len(j)/N
            
            break
    
    return centroid_center, centroid_cov, centroid_weight


def gaussian(mean, cov, x) :
    
    k = len(mean)
    eps = 0.0001
    epow = -0.5 * (x-mean).T @ np.linalg.inv(cov+eps*np.eye(k)) @ (x-mean)
    prob = (2 * np.pi)**(-0.5*k) * np.linalg.det(cov)**(-0.5) * np.exp(epow)
    
    return prob


class MixtureGaussian:
    def __init__(self, cluster, init_mean, init_cov, init_weight, eps=0.0001):
        self.cluster = cluster
        self.mu = init_mean
        self.cov = init_cov
        self.weight = init_weight
        self.gamma = []
        
        self.eps = eps
        self.N = len(cluster)
        self.dim = len(init_mean[0])
        self.n_cluster = len(init_mean)
    
    
    def expectation(self):
        # calc responsibility, gamma of each data point regarding three clusters 
        gamma_tmp = []
        for pts in self.cluster :
            tmp = []
            
            for i, j, k in zip(self.mu, self.cov, self.weight) :
                weighted_prob = k * gaussian(i, j, pts)
                tmp.append(weighted_prob)

            gamma_tmp.append(np.array(tmp)/np.array(tmp).sum())

        self.gamma = np.array(gamma_tmp)

        
    def maximization(self):
        # update mu, cov and weight based on responsibility
        # soft correspondence
        members = self.gamma.sum(axis=0)
        
        # update weight
        self.weight = members/self.N
        # update mean
        self.mu = np.array([i @ self.cluster for i in self.gamma.T])/members.reshape((self.n_cluster, -1))
        # update cov
        for i in range(3) :
            gamma = self.gamma.T[i]
            mu = self.mu[i]
            tmp = np.zeros((self.dim, self.dim))
            
            for idx, x in enumerate(self.cluster) :
                tmp += gamma[idx] * np.outer(x-mu, x-mu) / members[i]
                
            self.cov[i] = tmp
            
        
    def fit(self):
        
        while True:
            prev = self.mu.copy()
            
            self.expectation()
            self.maximization()
            
            convergence = sum([np.linalg.norm(k-l) for k, l in zip(self.mu, prev)])
            
            if convergence < self.eps: break
                
        return self.mu, self.cov, self.weight


    def sample(self, n_of_seq):
        
        sample_idx = np.random.choice(self.n_cluster, size=n_of_seq, p=self.weight)
        freq_idx = Counter(sample_idx)
        sample = []
        
        for key, item in freq_idx.items():
            new_elem = mv(self.mu[key], self.cov[key]).rvs(item)
            sample += list(new_elem)

        sample = np.array(sample)
        sort_idx = np.argsort(sample.T[0])

        sample = sample[sort_idx]
        
        return sample