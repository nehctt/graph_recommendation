import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from data_loader import MovieLensDataset
from models import GCN, GAT
from train import train_model, compute_recall_at_k

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Train GCN or GAT model for movie recommendations')
    
    # Model selection
    parser.add_argument('--model', type=str, default='gcn', choices=['gcn', 'gat'],
                      help='Model to train (gcn/gat)')
    
    # Dataset and model parameters
    parser.add_argument('--data', type=str, default='ml-100k',
                      help='Path to the MovieLens dataset')
    parser.add_argument('--dim', type=int, default=64,
                      help='Embedding dimension')
    parser.add_argument('--layers', type=int, default=2,
                      help='Number of GCN/GAT layers')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=100,
                      help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                      help='Learning rate')
    parser.add_argument('--batch', type=int, default=4096,
                      help='Batch size for training')
    parser.add_argument('--device', type=str, default='mps',
                      help='Device to use (mps/cuda/cpu)')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device(args.device)
    print(f'Using device: {device}')
    
    # Load data
    dataset = MovieLensDataset(args.data, test_size=0.2)
    adj_matrix = dataset.get_adjacency_matrix()
    train_data = dataset.get_train_data()
    test_data = dataset.get_test_data()
    user_mapping = dataset.get_user_mapping()
    item_mapping = dataset.get_item_mapping()
    
    # Initialize model
    n_users = dataset.get_num_users()
    n_items = dataset.get_num_items()
    
    if args.model == 'gcn':
        model = GCN(n_users, n_items, 
                   embedding_dim=args.dim, 
                   n_layers=args.layers)
    else:  # GAT
        model = GAT(n_users, n_items, 
                   embedding_dim=args.dim, 
                   n_layers=args.layers)
    
    # Train model
    print(f"\nTraining {args.model.upper()} model...")
    recall = train_model(model, adj_matrix, train_data, test_data, 
                        user_mapping, item_mapping, 
                        n_epochs=args.epochs,
                        lr=args.lr,
                        batch_size=args.batch,
                        device=device)
    print(f"Best {args.model.upper()} Recall@20: {recall:.4f}")

if __name__ == "__main__":
    main() 