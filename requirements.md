# requirements.txt

# --- Configuration Flags ---
# Use the PyTorch index as an *additional* source, not the only source.
--extra-index-url https://download.pytorch.org/whl/cu121
# Find PyG links in its specific repository.
--find-links https://data.pyg.org/whl/torch-2.3.0+cu121.html

# --- Base Libraries (from PyPI) ---
streamlit
pandas
numpy<2.0
scikit-learn
requests
tqdm

# --- PyTorch Libraries (from the extra index and find-links) ---
torch==2.3.0+cu121
torch_geometric==2.5.0
torch_sparse
torch_scatter
torch_cluster
