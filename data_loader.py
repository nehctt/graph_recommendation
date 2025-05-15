import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

class MovieLensDataset:
    def __init__(self, data_path, test_size=0.2, random_state=42):
        self.data_path = data_path
        self.test_size = test_size
        self.random_state = random_state
        
        # Load and preprocess data
        self.load_data()
        self.create_mappings()
        self.create_adjacency_matrix()
        
    def load_data(self):
        # Load ratings data (MovieLens-100K format)
        ratings = pd.read_csv(f"{self.data_path}/u.data", sep='\t', 
                            names=['userId', 'movieId', 'rating', 'timestamp'])
        
        # Keep only necessary columns
        self.ratings = ratings[['userId', 'movieId', 'rating']]
        
        # Split into train and test
        self.train_ratings, self.test_ratings = train_test_split(
            self.ratings, test_size=self.test_size, random_state=self.random_state
        )
        
    def create_mappings(self):
        # Create user and item mappings
        self.user_mapping = {user: idx for idx, user in enumerate(self.ratings['userId'].unique())}
        self.item_mapping = {item: idx for idx, item in enumerate(self.ratings['movieId'].unique())}
        
        self.n_users = len(self.user_mapping)
        self.n_items = len(self.item_mapping)
        
    def create_adjacency_matrix(self):
        # Create adjacency matrix for the graph
        self.adj_matrix = torch.zeros((self.n_users + self.n_items, self.n_users + self.n_items))
        
        # Add edges between users and items
        for _, row in self.train_ratings.iterrows():
            user_idx = self.user_mapping[row['userId']]
            item_idx = self.item_mapping[row['movieId']] + self.n_users
            
            # Add bidirectional edges
            self.adj_matrix[user_idx, item_idx] = 1
            self.adj_matrix[item_idx, user_idx] = 1
            
    def get_train_data(self):
        return self.train_ratings
    
    def get_test_data(self):
        return self.test_ratings
    
    def get_adjacency_matrix(self):
        return self.adj_matrix
    
    def get_num_users(self):
        return self.n_users
    
    def get_num_items(self):
        return self.n_items
    
    def get_user_mapping(self):
        return self.user_mapping
    
    def get_item_mapping(self):
        return self.item_mapping 