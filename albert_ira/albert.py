import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
import os
from torch import nn
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score
from gensim.models import KeyedVectors

# Constants
MODEL_NAME = "albert-base-v2"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load ALBERT tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
albert_model = AutoModel.from_pretrained(MODEL_NAME).to(device)

# Load GloVe embeddings from a file
def load_glove_model(glove_file_path):
    """Load GloVe embeddings into a dictionary."""
    embedding_dict = {}
    with open(glove_file_path, 'r', encoding="utf-8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = torch.tensor(np.asarray(values[1:], dtype='float32'))
            embedding_dict[word] = vector.to(device)  # Move to device only if needed
    return embedding_dict

# Load FastText embeddings from a file
def load_fasttext_model(fasttext_file_path):
    """Load FastText embeddings into a dictionary."""
    model = KeyedVectors.load_word2vec_format(fasttext_file_path, binary=False)
    return {word: torch.tensor(model[word]).to(device) for word in model.index_to_key}

# Get ALBERT embedding for a given text
def get_albert_embedding(text):
    """Generate ALBERT embedding for a given text using GPU."""
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=256).to(device)
    with torch.no_grad():
        outputs = albert_model(**inputs)
    return outputs.last_hidden_state[:, 0, :].cpu().numpy()

# Get word embedding (GloVe or FastText) for a given text
def get_word_embedding(text, embedding_dict):
    """Generate word embedding by averaging embeddings for each word in the text."""
    words = text.lower().split()
    vectors = [embedding_dict[word] for word in words if word in embedding_dict]
    if vectors:
        return torch.mean(torch.stack(vectors), dim=0).cpu().numpy()
    return np.zeros(300)

# Create combined embedding from ALBERT and optional GloVe/FastText embeddings
def create_combined_embedding(text, embedding_type=None, glove_model=None, fasttext_model=None):
    """Combine ALBERT embedding with optional GloVe or FastText embeddings for a given text."""
    albert_emb = get_albert_embedding(text).flatten()
    if embedding_type == "glove":
        additional_emb = get_word_embedding(text, glove_model)
    elif embedding_type == "fasttext":
        additional_emb = get_word_embedding(text, fasttext_model)
    else:
        additional_emb = np.array([])  # No additional embedding

    combined_emb = np.concatenate([albert_emb, additional_emb]) if additional_emb.size > 0 else albert_emb
    combined_emb_size = combined_emb.size  # Store the size of the combined embedding

    return combined_emb, combined_emb_size  # Return both the combined embedding and its size

# Define a regression model for predicting scores
class RegressionModel(nn.Module):
    """Simple feedforward neural network for regression with dropout for regularization."""
    def __init__(self, input_shape):
        super(RegressionModel, self).__init__()
        self.fc1 = nn.Linear(input_shape, 256)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 128)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

# Load a trained model from file
def load_model(model_path, input_shape):
    """Load a regression model from a saved file and set it to evaluation mode."""
    model = RegressionModel(input_shape).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model