import os
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score
from gensim.models import KeyedVectors
from sklearn.metrics import mean_squared_error, f1_score, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns

def load_glove_model(glove_file_path):
    """
    Load GloVe embeddings into a dictionary.
    :param glove_file_path: Path to the GloVe embedding file.
    :return: Dictionary with word-to-vector mappings.
    """
    embedding_dict = {}
    with open(glove_file_path, 'r', encoding="utf-8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = torch.tensor(np.asarray(values[1:], dtype='float32'))
            embedding_dict[word] = vector.to(device)
    return embedding_dict

def load_fasttext_model(fasttext_file_path):
    """
    Load FastText embeddings into a dictionary.
    :param fasttext_file_path: Path to the FastText embedding file.
    :return: Dictionary with word-to-vector mappings.
    """
    model = KeyedVectors.load_word2vec_format(fasttext_file_path, binary=False)
    return {word: torch.tensor(model[word]).to(device) for word in model.index_to_key}


class MultiTaskDependent(nn.Module):
    """
    A multitask neural network model for predicting classification scores
    for various essay attributes such as language, prompt adherence, etc.
    """
    def __init__(self, input_shape, num_classes):
        super(MultiTaskDependent, self).__init__()
        # Shared layers
        self.fc1 = nn.Linear(input_shape, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(0.5)

        # Task-specific heads
        self.language_head = nn.Linear(128, num_classes['language'])
        self.prompt_adherence_head = nn.Linear(128, num_classes['prompt_adherence'])
        self.narrativity_head = nn.Linear(128, num_classes['narrativity'])

    def forward(self, x):
        x = torch.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        x = torch.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)

        language_output = self.language_head(x)
        prompt_adherence_output = self.prompt_adherence_head(x)
        narrativity_output = self.narrativity_head(x)

        return language_output, prompt_adherence_output, narrativity_output

    def compute_uncertainty_loss(self, loss_language, loss_prompt_adherence, loss_narrativity):
        """
        Compute dynamically weighted loss using task uncertainty parameters.
        :return: Total weighted loss.
        """
        # Dynamic weighting of losses based on task uncertainty
        language_precision = torch.exp(-self.task_uncertainty[1])  # Precision for language
        prompt_adherence_precision = torch.exp(-self.task_uncertainty[1])  # Precision for word choice
        narrativity_precision = torch.exp(-self.task_uncertainty[1])  # Precision for sentence fluency

        # Weighted loss computation
        loss = (
            language_precision * loss_language + self.task_uncertainty[1] +
            prompt_adherence_precision * loss_prompt_adherence + self.task_uncertainty[1] +
            narrativity_precision * loss_narrativity + self.task_uncertainty[1]
        )
        return loss

    def compute_loss(self, pred_language, pred_prompt_adherence, pred_narrativity,
                     y_language, y_prompt_adherence, y_narrativity):
        """
        Compute the total loss across all tasks using CrossEntropyLoss.
        :return: Combined loss.
        """
        criterion = nn.CrossEntropyLoss()  # Standard cross-entropy loss

        # Compute individual losses for each task
        mse_loss_language = criterion(pred_language, y_language)
        mse_loss_prompt_adherence = criterion(pred_prompt_adherence, y_prompt_adherence)
        mse_loss_narrativity = criterion(pred_narrativity, y_narrativity)

        # Combine losses from all tasks
        total_loss = (
            mse_loss_language + 
            mse_loss_prompt_adherence + 
            mse_loss_narrativity
        )
        
        return total_loss

class MultiTaskArgumentative(nn.Module):
    """
    A multitask neural network model for predicting classification scores
    for various essay attributes such as organization, word choice, etc.
    """
    def __init__(self, input_shape, num_classes=7):
        super(MultiTaskArgumentative, self).__init__()
        # Shared layers
        self.fc1 = nn.Linear(input_shape, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(0.5)

        # Create separate classification heads for each feature
        self.organization_head = nn.Linear(128, num_classes)
        self.word_choice_head = nn.Linear(128, num_classes)
        self.sentence_fluency_head = nn.Linear(128, num_classes)
        self.conventions_head = nn.Linear(128, num_classes)

        # Optional task uncertainty parameter
        self.task_uncertainty = nn.Parameter(torch.tensor([0.0, 0.0]), requires_grad=True)        

    def forward(self, x):
        """
        Forward pass through the shared and task-specific layers.
        :param x: Input tensor
        :return: Outputs for each task
        """
        x = torch.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        x = torch.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)

        # Output for each feature
        organization_output = self.organization_head(x)
        word_choice_output = self.word_choice_head(x)
        sentence_fluency_output = self.sentence_fluency_head(x)
        conventions_output = self.conventions_head(x)

        return organization_output, word_choice_output, sentence_fluency_output, conventions_output

    def compute_uncertainty_loss(self, loss_organization, loss_word_choice, loss_sentence_fluency, loss_conventions):
        """
        Compute dynamically weighted loss using task uncertainty parameters.
        :return: Total weighted loss
        """
        organization_precision = torch.exp(-self.task_uncertainty[1])
        word_choice_precision = torch.exp(-self.task_uncertainty[1])
        sentence_fluency_precision = torch.exp(-self.task_uncertainty[1])
        conventions_precision = torch.exp(-self.task_uncertainty[1])

        # Weighted loss calculation
        loss = (organization_precision * loss_organization + self.task_uncertainty[1]) + \
                (word_choice_precision * loss_word_choice + self.task_uncertainty[1]) + \
                (sentence_fluency_precision * loss_sentence_fluency + self.task_uncertainty[1]) + \
                (conventions_precision * loss_conventions + self.task_uncertainty[1])
        
        return loss

    def compute_loss(self, pred_organization, pred_word_choice, pred_sentence_fluency, pred_conventions,
                        y_organization, y_word_choice, y_sentence_fluency, y_conventions) :

        """
        Compute total loss across all tasks.
        :return: Combined loss
        """
        criterion = nn.CrossEntropyLoss()
        mse_loss_organization = criterion(pred_organization, y_organization)
        mse_loss_word_choice = criterion(pred_word_choice, y_word_choice)
        mse_loss_sentence_fluency = criterion(pred_sentence_fluency, y_sentence_fluency)
        mse_loss_conventions = criterion(pred_conventions, y_conventions)

        total_loss = mse_loss_organization + mse_loss_word_choice + mse_loss_sentence_fluency + mse_loss_conventions
        
        return total_loss

class MultiTaskModel(nn.Module):
    """
    A multitask model for predicting regression (score).
    Incorporates shared layers for feature extraction and a task-specific regression head.
    """
    def __init__(self, input_shape):
        """
        Initialize the model layers.

        Args:
            input_shape (int): Size of the input features.
        """
        super(MultiTaskModel, self).__init__()
        # Shared layers for feature extraction
        self.fc1 = nn.Linear(input_shape, 256)  # Fully connected layer with 256 output units
        self.bn1 = nn.BatchNorm1d(256)         # Batch normalization to stabilize training
        self.dropout1 = nn.Dropout(0.5)        # Dropout for regularization
        self.fc2 = nn.Linear(256, 128)         # Fully connected layer with 128 output units
        self.bn2 = nn.BatchNorm1d(128)         # Batch normalization
        self.dropout2 = nn.Dropout(0.5)        # Dropout

        # Task-specific head for regression
        self.regression_head = nn.Linear(128, 1)  # Outputs a single regression value (score)

        # Learnable task uncertainty parameters (optional for weighted losses)
        self.task_uncertainty = nn.Parameter(torch.tensor([0.0, 0.0]), requires_grad=True)

    def forward(self, x):
        """
        Forward pass through the shared layers and the regression head.

        Returns:
            Tensor: Predicted score.
        """
        # Shared layers for feature extraction
        x = torch.relu(self.bn1(self.fc1(x)))  # First fully connected layer with ReLU activation
        x = self.dropout1(x)                  # Dropout for regularization
        x = torch.relu(self.bn2(self.fc2(x))) # Second fully connected layer with ReLU
        x = self.dropout2(x)                  # Dropout

        # Task-specific regression output
        score_output = self.regression_head(x)
        return score_output

    def compute_uncertainty_loss(self, loss_score):
        """
        Compute the weighted uncertainty loss for the regression task.

        Returns:
            Tensor: Weighted loss.
        """
        precision1 = torch.exp(-self.task_uncertainty[0])  # Precision for score task
        loss = (precision1 * loss_score + self.task_uncertainty[0])  # Weighted loss for score task
        return loss

    def compute_loss(self, pred_score, y_score):
        """
        Compute the loss for the regression task.

        Returns:
            Tensor: Mean Squared Error loss.
        """
        mse_loss = nn.MSELoss()(pred_score, y_score)  # MSE for regression
        return mse_loss

class LabelSmoothingCrossEntropy(nn.Module):
    """
    Custom loss function that incorporates label smoothing into the standard CrossEntropyLoss.
    """
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing  # Degree of label smoothing

    def forward(self, pred, target):
        """
        Compute the label-smoothed cross-entropy loss.
        :param pred: Predictions (logits) from the model.
        :param target: Ground truth labels.
        :return: Smoothed cross-entropy loss.
        """
        log_probs = F.log_softmax(pred, dim=-1)  # Convert logits to log probabilities

        # Compute negative log likelihood loss
        nll_loss = -log_probs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)  # Remove extra dimension

        # Compute the smoothed loss
        smooth_loss = -log_probs.mean(dim=-1)

        # Combine the two losses
        return (1 - self.smoothing) * nll_loss + self.smoothing * smooth_loss

def get_albert_embedding(text):
    """
    Generate ALBERT embeddings for a given text.

    Args:
        text (str): Input text.

    Returns:
        numpy.ndarray: The embedding vector from ALBERT's last hidden state.
    """
    # Tokenize the input text and send to the device (CPU/GPU)
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=256).to(device)
    
    # Generate embeddings without computing gradients
    with torch.no_grad():
        outputs = albert_model(**inputs)
    
    # Extract the [CLS] token embedding from the last hidden state
    return outputs.last_hidden_state[:, 0, :].cpu().numpy()

def get_word_embedding(text, embedding_dict):
    """
    Generate word embeddings for a given text using a pre-trained embedding dictionary.

    Args:
        text (str): Input text.
        embedding_dict (dict): Pre-trained word embedding dictionary (e.g., GloVe or FastText).

    Returns:
        numpy.ndarray: The average word embedding vector for the input text.
    """
    # Split text into words and fetch embeddings for each word if available
    words = text.lower().split()
    vectors = [embedding_dict[word] for word in words if word in embedding_dict]
    
    # Compute the average embedding if vectors are found; otherwise return a zero vector
    if vectors:
        return torch.mean(torch.stack(vectors), dim=0).cpu().numpy()
    return np.zeros(300)  # Default to 300 dimensions

def create_attention_based_embedding(albert_emb, additional_emb):
    """
    Create an attention-based fused embedding from ALBERT and additional embeddings.

    Args:
        albert_emb (torch.Tensor): ALBERT embedding vector.
        additional_emb (torch.Tensor): Additional embedding vector (e.g., GloVe or FastText).

    Returns:
        torch.Tensor: Fused embedding based on learned attention weights.
    """
    # Ensure both embeddings have the same shape
    if albert_emb.shape != additional_emb.shape:
        additional_emb = torch.nn.Linear(additional_emb.shape[0], albert_emb.shape[0]).to(albert_emb.device)(additional_emb)
    
    # Combine embeddings into a tensor stack
    combined_emb = torch.cat([albert_emb.unsqueeze(0), additional_emb.unsqueeze(0)], dim=0)
    
    # Learn attention weights dynamically
    attention_weights = torch.nn.Parameter(torch.tensor([0.5, 0.5], device=albert_emb.device), requires_grad=True)
    attention_scores = F.softmax(attention_weights, dim=0)
    
    # Compute the fused embedding as a weighted sum
    fused_embedding = attention_scores[0] * albert_emb + attention_scores[1] * additional_emb
    return fused_embedding


def create_combined_embedding(text, embedding_type=None, _glove_model=None, _fasttext_model=None):
    """
    Generate a combined embedding by fusing ALBERT and an additional embedding (GloVe/FastText).

    Args:
        text (str): Input text.
        embedding_type (str, optional): Type of additional embedding ("glove" or "fasttext"). Default is None.
        _glove_model (dict, optional): GloVe embedding dictionary. Required if embedding_type is "glove".
        _fasttext_model (dict, optional): FastText embedding dictionary. Required if embedding_type is "fasttext".

    Returns:
        tuple: Combined embedding as a numpy array and its size.
    """
    # Get ALBERT embedding
    albert_emb = get_albert_embedding(text).flatten()

    # Get the additional embedding based on the specified type
    if embedding_type == "glove":
        additional_emb = get_word_embedding(text, _glove_model)
    elif embedding_type == "fasttext":
        additional_emb = get_word_embedding(text, _fasttext_model)
    else:
        additional_emb = np.array([])

    # Convert ALBERT embedding to tensor
    albert_emb_tensor = torch.tensor(albert_emb, dtype=torch.float32).to(device)

    # Combine ALBERT and additional embeddings, ensuring equal size
    if additional_emb.size != 0:
        additional_emb_tensor = torch.tensor(additional_emb, dtype=torch.float32).to(device)
        if additional_emb_tensor.size(0) > albert_emb_tensor.size(0):
            additional_emb_tensor = additional_emb_tensor[:albert_emb_tensor.size(0)]
        elif additional_emb_tensor.size(0) < albert_emb_tensor.size(0):
            padding_size = albert_emb_tensor.size(0) - additional_emb_tensor.size(0)
            additional_emb_tensor = F.pad(additional_emb_tensor, (0, padding_size))
        combined_emb = torch.cat([albert_emb_tensor, additional_emb_tensor], dim=0)
    else:
        combined_emb = albert_emb_tensor

    # Return the combined embedding and its size
    return combined_emb.cpu().numpy(), combined_emb.size(0)