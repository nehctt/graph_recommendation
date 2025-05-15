import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

def compute_recall_at_k(user_emb, item_emb, test_data, user_mapping, item_mapping, k=20, device='mps'):
    # Convert test data to tensor format
    test_users = torch.tensor([user_mapping[user] for user in test_data['userId'].unique()], device=device)
    
    # Compute cosine similarity between users and items
    user_emb_norm = F.normalize(user_emb, p=2, dim=1)
    item_emb_norm = F.normalize(item_emb, p=2, dim=1)
    similarity = torch.mm(user_emb_norm, item_emb_norm.t())
    
    # Get top-k items for each user
    _, top_k_items = torch.topk(similarity, k=k, dim=1)
    
    # Compute recall@k
    recall_sum = 0
    for i, user in enumerate(test_users):
        user_items = set(test_data[test_data['userId'] == user.cpu().item()]['movieId'].values)
        user_items = set(item_mapping[item] for item in user_items)
        top_k = set(top_k_items[i].cpu().numpy())
        if len(user_items) > 0:
            recall_sum += len(user_items.intersection(top_k)) / len(user_items)
    
    return recall_sum / len(test_users)

def train_model(model, adj_matrix, train_data, test_data, user_mapping, item_mapping, 
                n_epochs=100, lr=0.001, batch_size=4096, device='mps'):
    model = model.to(device)
    adj_matrix = adj_matrix.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Use BPR loss for better ranking
    def bpr_loss(pos_scores, neg_scores):
        return -torch.mean(torch.log(torch.sigmoid(pos_scores - neg_scores)))
    
    # Prepare training data
    pos_edges = train_data[['userId', 'movieId']].values
    pos_users = torch.tensor([user_mapping[user] for user in pos_edges[:, 0]], device=device)
    pos_items = torch.tensor([item_mapping[item] for item in pos_edges[:, 1]], device=device)
    
    # Create dataset and dataloader
    train_dataset = TensorDataset(pos_users, pos_items)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    best_recall = 0
    for epoch in range(n_epochs):
        model.train()
        total_loss = 0
        n_batches = 0
        
        for batch_users, batch_items in train_loader:
            optimizer.zero_grad()
            
            # Forward pass
            user_emb, item_emb = model(adj_matrix)
            
            # Sample negative items for each user in batch
            neg_items = []
            for user_idx in batch_users.cpu().numpy():
                # Get items that the user hasn't interacted with
                user_id = list(user_mapping.keys())[list(user_mapping.values()).index(user_idx)]
                user_pos_items = set(train_data[train_data['userId'] == user_id]['movieId'].values)
                all_items = set(item_mapping.keys())
                neg_candidates = list(all_items - user_pos_items)
                if len(neg_candidates) > 0:
                    neg_item = np.random.choice(neg_candidates)
                    neg_items.append(item_mapping[neg_item])
                else:
                    neg_items.append(np.random.randint(0, len(item_mapping)))
            neg_items = torch.tensor(neg_items, device=device)
            
            # Compute scores for current batch
            pos_scores = torch.sum(user_emb[batch_users] * item_emb[batch_items], dim=1)
            neg_scores = torch.sum(user_emb[batch_users] * item_emb[neg_items], dim=1)
            
            # Compute loss for current batch
            loss = bpr_loss(pos_scores, neg_scores)
            
            # Add L2 regularization
            l2_reg = torch.tensor(0., device=device)
            for param in model.parameters():
                l2_reg += torch.norm(param)
            loss += 0.01 * l2_reg
            
            # Backward pass and update
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
        
        # Evaluate
        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                user_emb, item_emb = model(adj_matrix)
                recall = compute_recall_at_k(user_emb, item_emb, test_data, 
                                          user_mapping, item_mapping, k=20,
                                          device=device)
                avg_loss = total_loss / n_batches
                print(f'Epoch {epoch+1}/{n_epochs}, Loss: {avg_loss:.4f}, Recall@20: {recall:.4f}')
                
                if recall > best_recall:
                    best_recall = recall
                    torch.save(model.state_dict(), f'best_model_{model.__class__.__name__}.pt')
    
    return best_recall 