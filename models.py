import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        
    def forward(self, x, adj):
        # Normalize adjacency matrix
        adj = adj + torch.eye(adj.size(0)).to(adj.device)  # Add self-loops
        deg = torch.sum(adj, dim=1)
        deg_inv_sqrt = torch.pow(deg, -0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        adj_normalized = adj * deg_inv_sqrt.unsqueeze(1) * deg_inv_sqrt.unsqueeze(0)
        
        # GCN layer
        return self.linear(torch.mm(adj_normalized, x))

class GCN(nn.Module):
    def __init__(self, n_users, n_items, embedding_dim=64, n_layers=2):
        super(GCN, self).__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        
        # Initialize embeddings
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)
        
        # Initialize GCN layers
        self.gcn_layers = nn.ModuleList([
            GCNLayer(embedding_dim, embedding_dim) for _ in range(n_layers)
        ])
        
        # Initialize embeddings
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)
        
    def forward(self, adj):
        # Get initial embeddings
        user_emb = self.user_embedding(torch.arange(self.n_users, device=adj.device))
        item_emb = self.item_embedding(torch.arange(self.n_items, device=adj.device))
        
        # Concatenate user and item embeddings
        x = torch.cat([user_emb, item_emb], dim=0)
        
        # Apply GCN layers
        for gcn_layer in self.gcn_layers:
            x = F.relu(gcn_layer(x, adj))
            
        # Split embeddings back into users and items
        user_emb, item_emb = torch.split(x, [self.n_users, self.n_items], dim=0)
        
        return user_emb, item_emb

class SimpleGATLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.6, alpha=0.2):
        super(SimpleGATLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Linear transformation
        self.W = nn.Linear(in_features, out_features)
        
        # Attention mechanism
        self.a = nn.Parameter(torch.zeros(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data)
        
        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(alpha)
        
    def forward(self, h, adj):
        # Apply linear transformation
        Wh = self.W(h)  # [N, out_features]
        
        # Calculate attention scores
        a_input = torch.cat([Wh.repeat(1, Wh.size(0)).view(Wh.size(0) * Wh.size(0), -1),
                           Wh.repeat(Wh.size(0), 1)], dim=1).view(Wh.size(0), Wh.size(0), 2 * self.out_features)
        
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))
        
        # Mask attention scores with adjacency matrix
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = self.dropout(attention)
        
        # Apply attention
        h_prime = torch.matmul(attention, Wh)
        
        return h_prime

class GAT(nn.Module):
    def __init__(self, n_users, n_items, embedding_dim=64, n_layers=2, dropout=0.6):
        super(GAT, self).__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        
        # Initial embeddings
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)
        
        # Initialize embeddings
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)
        
        # GAT layers - using simpler implementation
        self.gat_layers = nn.ModuleList([
            SimpleGATLayer(embedding_dim, embedding_dim, dropout) for _ in range(n_layers)
        ])
        
        # Output layer
        self.output_layer = nn.Linear(embedding_dim, embedding_dim)
        
    def forward(self, adj):
        # Get initial embeddings
        user_emb = self.user_embedding(torch.arange(self.n_users, device=adj.device))
        item_emb = self.item_embedding(torch.arange(self.n_items, device=adj.device))
        
        # Concatenate user and item embeddings
        x = torch.cat([user_emb, item_emb], dim=0)
        
        # Apply GAT layers
        for gat_layer in self.gat_layers:
            x = F.relu(gat_layer(x, adj))
        
        # Split back into user and item embeddings
        user_emb, item_emb = torch.split(x, [self.n_users, self.n_items], dim=0)
        
        # Apply output layer
        user_emb = self.output_layer(user_emb)
        item_emb = self.output_layer(item_emb)
        
        return user_emb, item_emb 