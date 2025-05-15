# Movie Recommendation System using Graph Neural Networks

This project implements a movie recommendation system using Graph Neural Networks (GNNs), specifically Graph Convolutional Networks (GCN) and Graph Attention Networks (GAT) on the MovieLens dataset.

## Features

- Implements both GCN and GAT models for movie recommendations
- Uses BPR (Bayesian Personalized Ranking) loss for better ranking
- Supports batch training for efficient learning
- Evaluates using Recall@20 metric
- Supports MPS (Metal Performance Shaders) for Mac GPU acceleration

## Requirements

- Python 3.8+
- PyTorch 2.0+
- pandas
- numpy
- scikit-learn

Install dependencies:
```bash
pip install -r requirements.txt
```

## Dataset

The project uses the MovieLens-100k dataset. To set up the dataset:

1. Download the MovieLens-100k dataset from [GroupLens](https://grouplens.org/datasets/movielens/100k/)
2. Extract the downloaded zip file
3. Place the extracted `ml-100k` directory in the project root
4. The directory structure should look like:
```
project_root/
├── ml-100k/
│   ├── u.data
│   ├── u.item
│   └── ...
├── train.py
├── models.py
└── ...
```

## Usage

### Training

Train a GCN model:
```bash
python train.py --model gcn --epochs 100 --dim 64 --layers 2 --batch 4096
```

Train a GAT model:
```bash
python train.py --model gat --epochs 100 --dim 64 --layers 2 --batch 4096
```

### Command Line Arguments

- `--model`: Model to train ('gcn' or 'gat')
- `--data`: Path to MovieLens dataset (default: 'ml-100k')
- `--dim`: Embedding dimension (default: 64)
- `--layers`: Number of GNN layers (default: 2)
- `--epochs`: Number of training epochs (default: 100)
- `--lr`: Learning rate (default: 0.001)
- `--batch`: Batch size for training (default: 4096)
- `--device`: Device to use ('mps', 'cuda', or 'cpu', default: 'mps')

## Model Architecture

### GCN (Graph Convolutional Network)
- Uses graph convolution layers to aggregate neighbor information
- Simple and efficient implementation
- Good baseline for graph-based recommendation

### GAT (Graph Attention Network)
- Uses attention mechanism to learn importance of neighbors
- More expressive than GCN
- Can capture complex user-item interactions

## Training Process

1. Data Loading:
   - Loads MovieLens dataset
   - Creates user-item interaction graph
   - Splits data into train/test sets

2. Model Training:
   - Uses BPR loss for ranking optimization
   - Implements batch training for efficiency
   - Saves best model based on Recall@20

3. Evaluation:
   - Uses Recall@20 as the evaluation metric
   - Computes cosine similarity between user and item embeddings
   - Recommends top-20 items for each user