# train.py (Corrected Version)

import pandas as pd
import torch
from torch import nn, optim
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from sklearn.preprocessing import LabelEncoder
import requests
from io import BytesIO
import zipfile
from tqdm import tqdm

# --- Configuration ---
RATING_THRESHOLD = 3.5
EMBEDDING_DIM = 64
NUM_LAYERS = 3
BATCH_SIZE = 4096
EPOCHS = 50
LEARNING_RATE = 0.0005
LAMBDA_REG = 1e-5

print("Starting model training process...")

# --- Data Loading and Preprocessing ---
def load_and_prep_data():
    print("Step 1/4: Downloading and preparing data...")
    url = 'https://files.grouplens.org/datasets/movielens/ml-latest-small.zip'
    response = requests.get(url)
    zip_file = zipfile.ZipFile(BytesIO(response.content))
    with zip_file.open('ml-latest-small/ratings.csv') as f:
        ratings_df = pd.read_csv(f)
    
    user_encoder = LabelEncoder()
    movie_encoder = LabelEncoder()
    ratings_df['userId_encoded'] = user_encoder.fit_transform(ratings_df['userId'])
    ratings_df['movieId_encoded'] = movie_encoder.fit_transform(ratings_df['movieId'])
    
    num_users = len(ratings_df['userId_encoded'].unique())
    num_movies = len(ratings_df['movieId_encoded'].unique())
    
    return ratings_df, num_users, num_movies

def create_edge_index(df, threshold):
    print("Step 2/4: Creating graph edge index...")
    user_ids = torch.LongTensor(df['userId_encoded'].values)
    movie_ids = torch.LongTensor(df['movieId_encoded'].values)
    ratings = torch.FloatTensor(df['rating'].values) # Use FloatTensor for ratings
    mask = ratings >= threshold
    return torch.stack([user_ids[mask], movie_ids[mask]], dim=0).long()

# --- Model Definition ---
class LightGCN(MessagePassing):
    def __init__(self, num_users, num_items, embedding_dim, K):
        super().__init__()
        self.num_users, self.num_items, self.embedding_dim, self.K = num_users, num_items, embedding_dim, K
        self.users_emb = nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.embedding_dim)
        self.items_emb = nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.embedding_dim)
        nn.init.normal_(self.users_emb.weight, std=0.1)
        nn.init.normal_(self.items_emb.weight, std=0.1)

    def forward(self, edge_index):
        adj_matrix = self.create_adjacency_matrix(edge_index)
        emb_0 = torch.cat([self.users_emb.weight, self.items_emb.weight])
        embs = [emb_0]
        for _ in range(self.K):
            embs.append(torch.sparse.mm(adj_matrix, embs[-1]))
        emb_final = torch.mean(torch.stack(embs, dim=1), dim=1)
        users_emb, items_emb = torch.split(emb_final, [self.num_users, self.num_items])
        return users_emb, self.users_emb.weight, items_emb, self.items_emb.weight

    def create_adjacency_matrix(self, edge_index):
        item_indices_adj = edge_index[1] + self.num_users
        full_edge_index = torch.cat([torch.stack([edge_index[0], item_indices_adj]), torch.stack([item_indices_adj, edge_index[0]])], dim=1)
        edge_index_norm, edge_weight_norm = gcn_norm(full_edge_index, add_self_loops=False)
        # ** DEPRECATION FIX HERE **
        return torch.sparse_coo_tensor(edge_index_norm, edge_weight_norm, torch.Size((self.num_users + self.num_items, self.num_users + self.num_items)))

def bpr_loss(users_emb_final, users_emb_0, pos_items_emb_0, neg_items_emb_0, lambda_val):
    pos_scores = torch.sum(users_emb_final * pos_items_emb_0, dim=1)
    neg_scores = torch.sum(users_emb_final * neg_items_emb_0, dim=1)
    
    loss = -torch.log(torch.sigmoid(pos_scores - neg_scores)).mean()
    
    reg_loss = lambda_val * (
        (users_emb_0**2).sum() +
        (pos_items_emb_0**2).sum() +
        (neg_items_emb_0**2).sum()
    )
    return loss + reg_loss

# --- Main Training Execution ---
if __name__ == "__main__":
    ratings_df, num_users, num_movies = load_and_prep_data()
    edge_index = create_edge_index(ratings_df, RATING_THRESHOLD)

    # Move edge_index to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    edge_index = edge_index.to(device)

    model = LightGCN(num_users, num_movies, embedding_dim=EMBEDDING_DIM, K=NUM_LAYERS).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"Step 3/4: Starting training for {EPOCHS} epochs...")
    
    # Get all user and positive item pairs
    users = edge_index[0]
    pos_items = edge_index[1]
    
    # Create indices for shuffling
    perm = torch.randperm(users.size(0))

    for epoch in range(EPOCHS):
        model.train()
        
        # Shuffle the data at the start of each epoch
        perm = torch.randperm(users.size(0))
        users_perm = users[perm]
        pos_items_perm = pos_items[perm]

        total_loss = 0
        pbar = tqdm(range(0, users.size(0), BATCH_SIZE), desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for i in pbar:
            # ** NEW SAMPLING LOGIC HERE **
            user_batch = users_perm[i:i+BATCH_SIZE]
            pos_item_batch = pos_items_perm[i:i+BATCH_SIZE]
            
            # Sample negative items randomly from the entire item set
            neg_item_batch = torch.randint(0, num_movies, (len(user_batch),), device=device)
            
            optimizer.zero_grad()
            
            users_emb, u_emb_0, items_emb, i_emb_0 = model(edge_index)
            
            # Get embeddings for the batch
            u_emb_final_batch = users_emb[user_batch]
            u_emb_0_batch = u_emb_0[user_batch]
            pos_i_emb_0_batch = i_emb_0[pos_item_batch]
            neg_i_emb_0_batch = i_emb_0[neg_item_batch]

            loss = bpr_loss(u_emb_final_batch, u_emb_0_batch, pos_i_emb_0_batch, neg_i_emb_0_batch, LAMBDA_REG)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / (len(users) / BATCH_SIZE)
        print(f"Epoch {epoch+1}/{EPOCHS} - Average Loss: {avg_loss:.4f}")

    print("Step 4/4: Saving trained model weights...")
    # Move model back to CPU before saving to ensure compatibility
    torch.save(model.to('cpu').state_dict(), 'lightgcn_model_weights.pt')
    print("\n✅ Training complete. Model weights saved to 'lightgcn_model_weights.pt'.")
    print("You can now run the Streamlit app with `streamlit run app.py`")
