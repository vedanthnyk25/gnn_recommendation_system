# 🎬 GNN Movie Recommendation System

This project demonstrates a movie recommendation system using a Light Graph Convolutional Network (LightGCN) with PyTorch Geometric, served via an interactive Streamlit application.

The application loads a pre-trained model for fast recommendations and includes an optional interface to re-train the model with different hyperparameters.

---

## ⚙️ Setup

It's highly recommended to use a Python virtual environment to avoid package conflicts.

1.  **Create a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

2.  **Install the required libraries:**
    ```bash
    pip install -r requirements.txt
    ```

---

## 🚀 Usage

Running the project is a two-step process:

### **Step 1: Train the Model**

First, you need to train the GNN model. This script will download the dataset, train the model, and save the learned weights to a file named `lightgcn_model_weights.pt`.

**This only needs to be done once.**

```bash
python train.py
```

### **Step 2: Run the Streamlit Application**
```bash
streamlit run app.py
```
