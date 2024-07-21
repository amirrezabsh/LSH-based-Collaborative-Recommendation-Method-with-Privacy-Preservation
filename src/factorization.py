import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix

def pearson_similarity(tuser_ratings, other_user_ratings):
    def cc_pearson(x, y, mean_x, mean_y):
        numerator = np.sum((x - mean_x) * (y - mean_y))
        denominator = np.sqrt(np.sum((x - mean_x) ** 2)) * np.sqrt(np.sum((y - mean_y) ** 2))
        return numerator / denominator if denominator != 0 else 0

    common_items = tuser_ratings.multiply(other_user_ratings).nonzero()[1]
    
    if len(common_items) <= 1:
        similarity = 0
    else:
        tuser_ratings = tuser_ratings[0, common_items].toarray().flatten()
        other_user_ratings = other_user_ratings[0, common_items].toarray().flatten()
        
        target_user_mean = np.mean(tuser_ratings)
        other_user_mean = np.mean(other_user_ratings)

        similarity = cc_pearson(tuser_ratings, other_user_ratings, target_user_mean, other_user_mean)
    
    return similarity

class MatrixFactorization:
    def __init__(self, ratings, n_factors=100, l_rate=0.01, alpha=0.01, n_iter=100):
        self.ratings = ratings
        self.n_users, self.n_items = ratings.shape
        self.non_zero_row_ind, self.non_zero_col_ind = ratings.nonzero()
        self.n_interac = len(self.non_zero_row_ind)
        self.ind_lst = list(range(self.n_interac))
        self.n_factors = n_factors
        self.l_rate = l_rate  # eta0 Constant that multiplies the update term
        self.alpha = alpha  # lambda Constant that multiplies the regularization term
        self.n_iter = n_iter
        self.errors_lst = []
        self.n_iter_no_change = 10
        self.verbose = True
        self.stop = False
        
    def initialize(self):
        self.now = time.time()

        # Initialize user & item vectors        
        self.user_vecs = np.random.normal(scale=0.01, size=(self.n_users, self.n_factors)).astype(np.float32)
        self.item_vecs = np.random.normal(scale=0.01, size=(self.n_items, self.n_factors)).astype(np.float32)

        self.evaluate_model(0)
    
    def get_k_neighbors(self, k, tuser):
        similarities = []
        tuser_ratings = tuser.tocsr()

        for i in range(self.ratings.shape[0]):
            user_ratings = self.ratings.getrow(i)
            if user_ratings.nnz > 0:
                similarity = pearson_similarity(tuser_ratings, user_ratings)
                similarities.append((i, similarity))

        similarities.sort(key=lambda x: x[1], reverse=True)
        
        if k < len(similarities):
            return similarities[:k]
        else:
            return similarities
    
    def predict(self, u, i):
        return np.dot(self.user_vecs[u], self.item_vecs[i])
        
    def update_params(self, error, u, i):
        # Update User and item Vectors
        self.user_vecs[u, :] += self.l_rate * (error * self.item_vecs[i, :] - self.alpha * self.user_vecs[u, :])
        self.item_vecs[i, :] += self.l_rate * (error * self.user_vecs[u, :] - self.alpha * self.item_vecs[i, :])
        
    def evaluate_model(self, epoch):
        tot_error = 0
        for index in self.ind_lst:
            u, i = self.non_zero_row_ind[index], self.non_zero_col_ind[index]
            pred_rat = self.predict(u, i)
            tot_error += self.loss(pred_rat, self.ratings[u, i], self.user_vecs[u], self.item_vecs[i])
        error = tot_error / self.n_interac
        self.errors_lst.append(error)
        if self.verbose: 
            print(f"---> Epoch {epoch}")
            temp = np.round(time.time() - self.now, 3)
            print(f"Total error {np.round(self.errors_lst[-1], 3)} ===> Total training time: {temp} seconds.")
        
        
    def loss(self, y_pred, y_true, user_vec, item_vec):
        return (y_true - y_pred) + self.alpha * (np.sum(user_vec ** 2) + np.sum(item_vec ** 2))
    
    def fit(self):
        self.initialize()
        for epoch in range(1, self.n_iter):
            np.random.shuffle(self.ind_lst)
            for index in self.ind_lst:
                u, i = self.non_zero_row_ind[index], self.non_zero_col_ind[index]
                pred_rat = self.predict(u, i)
                error = self.loss(pred_rat, self.ratings[u, i], self.user_vecs[u], self.item_vecs[i])
                self.update_params(error, u, i)
            self.evaluate_model(epoch)

        # self.plot_the_score()
