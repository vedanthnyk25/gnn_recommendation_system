# app.py

import streamlit as st
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from sklearn.preprocessing import LabelEncoder
import requests
from io import BytesIO
import zipfile
import os

# --- Page Configuration ---
st.set_page_config(page_title="GNN Movie Recommender", page_icon="🎬", layout="wide")

st.title("🎬 GNN-Powered Movie Recommendation System")
st.write("""
This app uses a pre-trained **Light Graph Convolutional Network (LightGCN)** to provide personalized movie recommendations.
Select a user from the dropdown below to see their top 10 recommended movies.
""")

# --- Model Definition (must match the training script) ---
class LightGCN(MessagePassing):
    def __init__(self, num_users, num_items, embedding_dim=64, K=3):
        super().__init__()
        self.num_users, self.num_items, self.embedding_dim, self.K = num_users, num_items, embedding_dim, K
        self.users_emb = nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.embedding_dim)
        self.items_emb = nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.embedding_dim)
        # Initialization is not critical here as we are loading weights, but good practice.
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
        return torch.sparse.FloatTensor(edge_index_norm, edge_weight_norm, torch.Size((self.num_users + self.num_items, self.num_users + self.num_items)))

# --- Caching Functions for Performance ---
@st.cache_data
def load_data():
    """Loads and preprocesses data."""
    url = 'https://files.grouplens.org/datasets/movielens/ml-latest-small.zip'
    response = requests.get(url)
    zip_file = zipfile.ZipFile(BytesIO(response.content))
    ratings_df = pd.read_csv(zip_file.open('ml-latest-small/ratings.csv'))
    movies_df = pd.read_csv(zip_file.open('ml-latest-small/movies.csv'))

    user_encoder = LabelEncoder()
    movie_encoder = LabelEncoder()
    ratings_df['userId_encoded'] = user_encoder.fit_transform(ratings_df['userId'])
    ratings_df['movieId_encoded'] = movie_encoder.fit_transform(ratings_df['movieId'])
    
    # Create mapping from encoded ID back to original movie ID for recommendations
    movie_id_map = dict(zip(ratings_df['movieId'], ratings_df['movieId_encoded']))
    
    return ratings_df, movies_df, user_encoder, movie_encoder

@st.cache_resource
def load_model(num_users, num_movies, weights_path='lightgcn_model_weights.pt'):
    """Loads the pre-trained model weights."""
    model = LightGCN(num_users, num_movies, embedding_dim=64, K=3)
    model.load_state_dict(torch.load(weights_path))
    model.eval()
    return model

# --- Recommendation Function ---
def get_recommendations(model, user_id, user_encoder, movie_encoder, ratings_df, movies_df, num_recs=10):
    user_id_encoded = user_encoder.transform([user_id])[0]
    
    # Create the full graph for inference
    user_ids_tensor = torch.LongTensor(ratings_df['userId_encoded'].values)
    movie_ids_tensor = torch.LongTensor(ratings_df['movieId_encoded'].values)
    edge_index = torch.stack([user_ids_tensor, movie_ids_tensor], dim=0)

    with torch.no_grad():
        users_emb, _, items_emb, _ = model(edge_index)
        user_embedding = users_emb[user_id_encoded]
        scores = torch.matmul(user_embedding, items_emb.T)
        
    rated_movies = ratings_df[ratings_df['userId'] == user_id]['movieId_encoded'].tolist()
    scores[rated_movies] = -np.inf # Exclude already rated movies
    
    top_k_indices = torch.topk(scores, num_recs).indices.tolist()
    
    # Map encoded movie IDs back to titles
    recommended_movie_ids = movie_encoder.inverse_transform(top_k_indices)
    recs_df = movies_df[movies_df['movieId'].isin(recommended_movie_ids)]
    
    # Preserve order
    recs_df['order'] = recs_df['movieId'].apply(lambda x: list(recommended_movie_ids).index(x))
    return recs_df.sort_values('order')[['title', 'genres']]

# --- Main App Logic ---
MODEL_WEIGHTS_FILE = 'lightgcn_model_weights.pt'

if not os.path.exists(MODEL_WEIGHTS_FILE):
    st.error(f"Model weights file not found: `{MODEL_WEIGHTS_FILE}`")
    st.info("Please run the training script first from your terminal: `python train.py`")
else:
    # Load all necessary data and models
    with st.spinner("Loading data and pre-trained model..."):
        ratings_df, movies_df, user_encoder, movie_encoder = load_data()
        num_users = len(user_encoder.classes_)
        num_movies = len(movie_encoder.classes_)
        model = load_model(num_users, num_movies, MODEL_WEIGHTS_FILE)

    st.success("Model loaded successfully!")
    
    # --- UI for Inference ---
    st.header("✨ Get Your Recommendations")
    
    user_list = sorted(ratings_df['userId'].unique())
    selected_user_id = st.selectbox("Select a User ID:", user_list)
    
    if st.button("Generate Recommendations", type="primary"):
        with st.spinner(f"Finding movies for User {selected_user_id}..."):
            recommendations = get_recommendations(model, selected_user_id, user_encoder, movie_encoder, ratings_df, movies_df)
        
        st.subheader(f"Top 10 Recommendations for User {selected_user_id}")
        st.dataframe(recommendations, use_container_width=True, hide_index=True)
