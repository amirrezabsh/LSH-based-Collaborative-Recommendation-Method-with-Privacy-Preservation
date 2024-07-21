import numpy as np
from scipy.sparse import lil_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def filter_users_by_ratings(ratings, min_ratings):
    # Count the number of ratings per user
    user_rating_counts = ratings['userId'].value_counts()
    
    # Get the list of users who have at least min_ratings ratings
    users_to_keep = user_rating_counts[user_rating_counts >= min_ratings].index
    
    # Filter the ratings DataFrame to include only these users
    filtered_ratings = ratings[ratings['userId'].isin(users_to_keep)]
    
    return filtered_ratings



def split_train_test(sparse_ratings, split_ratio=0.8, user_split_ratio=0.3):
    # Ensure reproducibility
    np.random.seed(42)
    
    # Find the number of users in the sparse matrix
    num_users = sparse_ratings.shape[0]
    
    # Generate random indices for each user
    idxs = np.arange(num_users)
    np.random.shuffle(idxs)
    
    # Calculate the split point
    split_point = int(split_ratio * num_users)
    
    # Get the training and testing indices
    test_idx = idxs[split_point:]
    
    # Create initial train and test sets using these indices
    train_sparse_ratings = sparse_ratings.copy().tolil()
    test_sparse_ratings = lil_matrix(sparse_ratings.shape)
    
    # Iterate over each user in the test set
    for user in test_idx:
        # Get the non-zero ratings indices for the current user
        user_ratings = sparse_ratings[user, :].nonzero()[1]
        
        # Calculate the number of ratings to move to the test set
        num_user_ratings = len(user_ratings)
        num_test_ratings = int(user_split_ratio * num_user_ratings)
        
        # Randomly select indices to move to the test set
        np.random.shuffle(user_ratings)
        test_ratings_indices = user_ratings[:num_test_ratings]
        
        # Move the selected ratings to the test set
        test_sparse_ratings[user, test_ratings_indices] = train_sparse_ratings[user, test_ratings_indices]
        train_sparse_ratings[user, test_ratings_indices] = 0
    
    # Convert back to csr_matrix for efficient arithmetic operations
    train_sparse_ratings = train_sparse_ratings.tocsr()
    test_sparse_ratings = test_sparse_ratings.tocsr()
    
    return train_sparse_ratings, test_sparse_ratings

def spy_plot(matrix, sample_size=None, title="Spy Plot of Sparse Matrix"):
    if sample_size:
        rows = np.random.choice(matrix.shape[0], sample_size, replace=False)
        cols = np.random.choice(matrix.shape[1], sample_size, replace=False)
        matrix = matrix[rows][:, cols]
    
    plt.figure(figsize=(10, 10))
    plt.spy(matrix, markersize=0.1)
    plt.title(title)
    plt.show()

def density_heatmap(matrix, bins=100, title="Density Heatmap of Sparse Matrix"):
    density = np.zeros((bins, bins))
    row_bin = matrix.shape[0] // bins
    col_bin = matrix.shape[1] // bins
    
    for i in range(bins):
        for j in range(bins):
            row_start, row_end = i * row_bin, (i + 1) * row_bin
            col_start, col_end = j * col_bin, (j + 1) * col_bin
            submatrix = matrix[row_start:row_end, col_start:col_end]
            density[i, j] = submatrix.nnz / (row_bin * col_bin)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(density, cmap='viridis')
    plt.title(title)
    plt.show()