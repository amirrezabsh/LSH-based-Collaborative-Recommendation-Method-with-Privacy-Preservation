from lsh import LSH
from factorization import MatrixFactorization
from scipy.sparse import csr_matrix, vstack
import numpy as np
import pandas as pd
from utils import filter_users_by_ratings, split_train_test
import time
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor

def insert_into_lsh(lsh, train_row, user_id):
    print(f"Inserting user {user_id}")
    lsh.insert(train_row)
    
class Model:
    def __init__(self, train_dataset: csr_matrix,tags_dataset, movies_dataset,num_hash_tables=6, num_hashes: int = 3, n_factors: int = 100, l_rate: float = 0.01, alpha: float = 0.01, n_iter: int = 100) -> None:
        self.n_factors = n_factors
        self.n_iters = n_iter
        self.num_ratings = train_dataset.shape[1]
        self.lshs = [LSH(num_hashes, self.num_ratings) for i in range(num_hash_tables)]
        self.train_dataset = train_dataset
        self.matrix_factorizations = {}
        self.tags_dataset = tags_dataset
        self.movies_dataset = movies_dataset

    def cal_diversity(self, tags1, tags2):
        intersection = tags1.intersection(tags2)
        union = tags1.union(tags2)
        iou = len(intersection) / len(union) if len(union) != 0 else 0
        return iou
        
    def diversify(self, item_index, recommendations):
        # Assuming self.tags_dataset is a DataFrame with 'movieId' and 'tag' columns
        print(item_index)
        try:
            titem_tags = self.tags_dataset.loc[self.tags_dataset['movieId'] == item_index, 'tag'].values[0]

            items_to_remove = []
            for i, _ in recommendations:
                if i != item_index:
                    oitem_tags = self.tags_dataset.loc[self.tags_dataset['movieId'] == item_index, 'tag'].values[0]
                    diversity = self.cal_diversity(titem_tags, oitem_tags)
                    
                    if diversity == 1:
                        items_to_remove.append(i)
                    
                        

            return items_to_remove
        except:
            return []
                
        
                
    


    def train(self):
        start_time = time.time()
        
        print("Building LSH")
        with ThreadPoolExecutor() as executor:
            futures = []
            for i in range(self.train_dataset.shape[0]):
                train_row = self.train_dataset.getrow(i)
                if train_row.nnz > 0:
                    for lsh in self.lshs:
                        futures.append(executor.submit(insert_into_lsh, lsh, train_row, i))
                        
            for future in futures:
                future.result()  # Ensure all insertions are complete
        
        duration = time.time() - start_time
        print(f"Finished Building LSH. Duration: {duration//60} mins and {duration%60} secs")
            
    def predict(self, item_index, nearest_neighbors, tuser_mean, neighbors):
        predicted_rating = 0
        similarities_sum = 0
        for user_index, similarity in nearest_neighbors:
            try:
                neighbor_ratings = neighbors.getrow(user_index)
                neighbor_rating = neighbor_ratings[0, item_index]
                if neighbor_rating != 0:
                    neighbor_mean = np.mean(neighbor_ratings.data)
                    predicted_rating += similarity * (neighbor_rating - neighbor_mean)
                    similarities_sum += abs(similarity)
            except:
                pass
        if predicted_rating != 0:
            predicted_rating /= similarities_sum
            predicted_rating += tuser_mean
        return predicted_rating
    
    def recommend(self, tuser_index: int, k_neighbors: int, k_recomms: int, dataset=None):
        # Get the target user's ratings
        if dataset is None:
            dataset = self.train_dataset
        tuser_ratings = dataset.getrow(tuser_index)
        tuser_mean = np.mean(tuser_ratings.data)
        
        # Find the hash bucket for the target user
        bucket_idx, neighbors = self.lsh.query(tuser_ratings)

        factorization = self.matrix_factorizations[bucket_idx]
        
        # Get k-nearest neighbors using Pearson similarity
        nearest_neighbors = factorization.get_k_neighbors(k_neighbors, tuser_ratings)

        # Predict ratings for the target user based on nearest neighbors
        recommendations = []
        for item_index in range(1, self.num_ratings):
            if tuser_ratings[0, item_index] == 0:
                predicted_rating = self.predict(item_index, nearest_neighbors, tuser_mean, neighbors)
                recommendations.append((item_index, predicted_rating))

        # Sort recommendations by predicted rating in descending order
        recommendations.sort(key=lambda x: x[1], reverse=True)
        
        items_to_remove = []
        for item, _ in recommendations:
            result = self.diversify(item, recommendations)
            items_to_remove.extend(result)
        
        # Remove items from recommendations
        recommendations = [(i, rating) for i, rating in recommendations if i not in items_to_remove]
        
        # Print recommendations after filtering
        print("Filtered Recommendations:")
        for item, rating in recommendations[:k_recomms]:
            movie_name = self.movies_dataset.loc[self.movies_dataset['movieId'] == item + 1, 'title'].values[0]
            print(f"Movie ID: {item}, Title: {movie_name}, Predicted Rating: {rating}")
        
        
        return recommendations[:k_recomms]
    
    
    def remove_redundant_rows(self, csr_matrix):
        seen = set()
        unique_rows = []
        for row in csr_matrix:
            row_tuple = tuple(row.indices)
            if row_tuple not in seen:
                seen.add(row_tuple)
                unique_rows.append(row)
        return vstack(unique_rows)
    
    def calculate_mae(self, test_dataset):
        """
        Calculate the Mean Absolute Error (MAE) on the test dataset.
        """
        k_neighbors = 5
        total_error = 0
        count = 0
        mae_values = []
        user_indices = []
    
        # Get the row sums
        row_sums = test_dataset.sum(axis=1)
        # Find rows where the sum is not zero
        nonzero_row_indices = np.where(row_sums != 0)[0]
        # Filter out zero rows
        filtered_matrix = test_dataset[nonzero_row_indices]
        soorat = 0
        recall_makhraj = 0
        precision_makhraj = 0
        
        print(f"#Users to test: {filtered_matrix.shape[0]}")
        
        for u in range(filtered_matrix.shape[0]):
            user_ratings = filtered_matrix.getrow(u)
            
            if user_ratings.nnz > 0:
                # Get the target user's ratings
                tuser_mean = np.mean(user_ratings.data)
                
                # Find the hash bucket for the target user
                # bucket_idx, neighbors = self.lsh.query(user_ratings)
                all_neighbors = []
                for lsh in self.lshs:
                    bucket_idx, neighbors = lsh.query(user_ratings)
                    all_neighbors.append(neighbors)
                    
                neighbors_union = vstack(all_neighbors)
                
                neighbors_union = self.remove_redundant_rows(neighbors_union)
                
                factorization = MatrixFactorization(neighbors_union, self.n_factors, n_iter=self.n_iters)
                factorization.fit()
                
                # Get k-nearest neighbors using Pearson similarity
                nearest_neighbors = factorization.get_k_neighbors(k_neighbors, user_ratings)
                
                recommendations = []
                for i in user_ratings.indices:
                    predicted_rating = self.predict(i, nearest_neighbors, tuser_mean, neighbors_union)
                    total_error += abs(predicted_rating - user_ratings[0, i])
                    count += 1
                    
                    recommendations.append((i, predicted_rating))
                    
                # Sort recommendations by predicted rating in descending order
                recommendations.sort(key=lambda x: x[1], reverse=True)
                
                items_to_remove = []
                for item, _ in recommendations:
                    result = self.diversify(item, recommendations)
                    items_to_remove.extend(result)
                
                # Remove items from recommendations
                recommendations = [(i, rating) for i, rating in recommendations if i not in items_to_remove]
                
                for _, recomm in recommendations:
                
                    if recomm > 0 and user_ratings[0, i] > 0:
                        soorat += 1
                    if recomm > 0:
                        precision_makhraj += 1
                    if user_ratings[0, i] > 0:
                        recall_makhraj += 1
                        
                mae = total_error / count if count > 0 else float('inf')
                print(f"Total error until User {u}: {mae}")
                
                mae_values.append(mae)
                user_indices.append(u)
        
        
        print(f"Precision: {soorat / precision_makhraj if precision_makhraj != 0 else 0}")
        print(f"Recall: {soorat / recall_makhraj if recall_makhraj != 0 else 0}")
        
        self.plot_mae(user_indices, mae_values)
        
        return mae
    
    def plot_mae(self, user_indices, mae_values):
        # Plot MAE over users
        plt.figure(figsize=(10, 6))
        plt.plot(user_indices, mae_values, label='MAE')
        plt.xlabel('User Index')
        plt.ylabel('MAE')
        plt.title('MAE over Users')
        plt.legend()
        plt.show()
    
if __name__ == "__main__":
    # Load the dataset file
    ratings = pd.read_csv('ml-20m/ratings.csv')
    
    try:
        del ratings['timestamp']
    except:
        pass
    

    # Set the minimum number of ratings threshold
    min_ratings_threshold = 10

    # Filter users with fewer ratings than the threshold
    ratings = filter_users_by_ratings(ratings, min_ratings_threshold)
    
    ratings = ratings.sample(frac=0.0001, random_state=42)
    
    # Normalize userId and movieId to start from 0
    ratings['userId'] -= 1
    ratings['movieId'] -= 1
    
    # Create a sparse matrix
    sparse_ratings = csr_matrix((ratings['rating'].astype(np.float32), 
                                (ratings['userId'], ratings['movieId'])))
    
    
    # Split the dataset
    train_sparse_ratings, test_sparse_ratings = split_train_test(sparse_ratings, split_ratio=0.8, user_split_ratio=0.7)

    tags = pd.read_csv('ml-20m/tags.csv')
    tags_per_movie = tags.groupby('movieId')['tag'].agg(lambda x: set(x)).reset_index()

    movies = pd.read_csv('ml-20m/movies.csv')
    
    model = Model(train_dataset=train_sparse_ratings, tags_dataset=tags_per_movie, movies_dataset=movies, num_hashes=8, n_factors=10, n_iter=5)    
    model.train()

    print("Calculating MAE")
    print(f"MAE: {model.calculate_mae(test_sparse_ratings)}")

    print("Rcommending to user 10")
    model.recommend(10, 10, 10)