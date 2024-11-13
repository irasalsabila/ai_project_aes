import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
import os
from torch import nn
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from gensim.models import KeyedVectors
from sklearn.metrics import f1_score, accuracy_score, mean_squared_error, cohen_kappa_score
from sklearn.preprocessing import KBinsDiscretizer
import torch.nn.functional as F
import requests
import os
from groq import Groq
import math
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # Ensure CUDA launches are synchronous for debugging

# Constants
MODEL_NAME = "albert-base-v2"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load ALBERT tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
albert_model = AutoModel.from_pretrained(MODEL_NAME).to(device)

# Initialize Groq client
client = Groq(api_key="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")

class RegressionModel(nn.Module):
    """A simple feedforward neural network for regression, with dropout for regularization."""
    
    def __init__(self, input_shape):
        super(RegressionModel, self).__init__()
        self.fc1 = nn.Linear(input_shape, 256)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 128)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x):
        """Forward pass through the network."""
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x
        
class MultiTaskModel(nn.Module):
    """
    A multitask model for predicting both regression (score) and classification (quality).
    This model can also predict other attributes relevant to essay quality.
    """
    def __init__(self, input_shape):
        super(MultiTaskModel, self).__init__()
        # Shared layers
        self.fc1 = nn.Linear(input_shape, 256)
        self.bn1 = nn.BatchNorm1d(256)  # Batch normalization
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(0.5)
        
        # Heads for different tasks
        self.regression_head = nn.Linear(128, 1)
        self.classification_head = nn.Linear(128, 3)  # 3 classes: low, medium, high
        self.essay_type_head = nn.Linear(128, 2)  # 3 types: argumentative, dependent, narrative

        # Learnable task uncertainty parameters
        self.task_uncertainty = nn.Parameter(torch.tensor([0.0, 0.0]), requires_grad=True)

        # Additional regression heads for other attributes (score between 0-10)
        self.content_head = nn.Linear(128, 1)  # Content
        self.organization_head = nn.Linear(128, 1)  # Organization
        self.word_choice_head = nn.Linear(128, 1)  # Word Choice
        self.sentence_fluency_head = nn.Linear(128, 1)  # Sentence Fluency
        self.conventions_head = nn.Linear(128, 1)  # Conventions
        self.language_head = nn.Linear(128, 1)  # Language
        self.prompt_adherence_head = nn.Linear(128, 1)  # Prompt Adherence
        self.narrativity_head = nn.Linear(128, 1)  # Narrativity
        # self.style_head = nn.Linear(128, 1)  # Style
        # self.voice_head = nn.Linear(128, 1)  # Voice
     
    def forward(self, x):
        """Forward pass for multitask prediction."""
        x = torch.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        x = torch.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        
        # Outputs for each task
        score_output = self.regression_head(x)
        quality_output = self.classification_head(x)
        essay_type_output = self.essay_type_head(x)
        
        # Additional regression outputs for the attributes
        content_output = self.content_head(x)
        organization_output = self.organization_head(x)
        word_choice_output = self.word_choice_head(x)
        sentence_fluency_output = self.sentence_fluency_head(x)
        conventions_output = self.conventions_head(x)
        language_output = self.language_head(x)
        prompt_adherence_output = self.prompt_adherence_head(x)
        narrativity_output = self.narrativity_head(x)
        # style_output = self.style_head(x)
        # voice_output = self.voice_head(x)

        return (score_output, quality_output, essay_type_output,
                content_output, organization_output, word_choice_output,
                sentence_fluency_output, conventions_output,
                language_output, prompt_adherence_output, narrativity_output)

    def compute_uncertainty_loss(self, loss_score, loss_quality):
        """
        Compute the total loss with task uncertainty weighting.
        
        Args:
            loss_score: Loss for the regression task (score prediction).
            loss_quality: Loss for the classification task (quality prediction).
            
        Returns:
            Weighted total loss based on task uncertainties.
        """
        precision1 = torch.exp(-self.task_uncertainty[0])
        precision2 = torch.exp(-self.task_uncertainty[1])
        
        loss = (precision1 * loss_score + self.task_uncertainty[0]) + \
            (precision2 * loss_quality + self.task_uncertainty[1])
        
        return loss
        
    def compute_loss(self, pred_score, pred_quality, pred_essay_type,
                     pred_content, pred_organization, pred_word_choice,
                     pred_sentence_fluency, pred_conventions,
                     y_score, y_quality, y_essay_type,
                     y_content, y_organization, y_word_choice,
                     y_sentence_fluency, y_conventions, y_language, y_prompt_adherence, y_narrativity) :
        """
        Calculate the combined loss for all tasks in the multitask model.
        
        Args:
            Predictions and ground truth values for the main score and various quality attributes.
            
        Returns:
            Total loss calculated as a combination of regression and classification losses.
        """

        mse_loss = nn.MSELoss()(pred_score, y_score)
        cross_entropy_loss_quality = nn.CrossEntropyLoss()(pred_quality, y_quality)
        cross_entropy_loss_essay_type = nn.CrossEntropyLoss()(pred_essay_type, y_essay_type)

        # MSE loss for the additional attributes
        mse_loss_content = nn.MSELoss()(pred_content, y_content)
        mse_loss_organization = nn.MSELoss()(pred_organization, y_organization)
        mse_loss_word_choice = nn.MSELoss()(pred_word_choice, y_word_choice)
        mse_loss_sentence_fluency = nn.MSELoss()(pred_sentence_fluency, y_sentence_fluency)
        mse_loss_conventions = nn.MSELoss()(pred_conventions, y_conventions)
        mse_loss_language = nn.MSELoss()(pred_language, y_language)
        mse_loss_prompt_adherence = nn.MSELoss()(pred_prompt_adherence, y_prompt_adherence)
        mse_loss_narrativity = nn.MSELoss()(pred_narrativity, y_narrativity)
        # mse_loss_style = nn.MSELoss()(pred_style, y_style)
        # mse_loss_voice = nn.MSELoss()(pred_voice, y_voice)

        # Sum all the losses for total loss
        total_loss = mse_loss + cross_entropy_loss_quality + cross_entropy_loss_essay_type + \
                     mse_loss_content + mse_loss_organization + mse_loss_word_choice + \
                     mse_loss_sentence_fluency + mse_loss_conventions, mse_loss_language, \
                     mse_loss_prompt_adherence, mse_loss_narrativity
        
        return total_loss

class LabelSmoothingCrossEntropy(nn.Module):
    """Cross-entropy loss with label smoothing for regularization."""
    
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing

    def forward(self, pred, target):
        """Forward pass with label smoothing."""
        log_probs = F.log_softmax(pred, dim=-1)
        nll_loss = -log_probs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -log_probs.mean(dim=-1)
        return (1 - self.smoothing) * nll_loss + self.smoothing * smooth_loss

def load_glove_model(glove_file_path):
    """Load GloVe embeddings from a file into a dictionary."""
    embedding_dict = {}
    with open(glove_file_path, 'r', encoding="utf-8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = torch.tensor(np.asarray(values[1:], dtype='float32'))
            embedding_dict[word] = vector.to(device)  # Move to device if necessary
    return embedding_dict

def load_fasttext_model(fasttext_file_path):
    """Load FastText embeddings from a file into a dictionary."""
    model = KeyedVectors.load_word2vec_format(fasttext_file_path, binary=False)
    return {word: torch.tensor(model[word]).to(device) for word in model.index_to_key}

def get_albert_embedding(text):
    """Generate ALBERT embeddings for the input text."""
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=256).to(device)
    with torch.no_grad():
        outputs = albert_model(**inputs)
    return outputs.last_hidden_state[:, 0, :].cpu().numpy()

def get_word_embedding(text, embedding_dict):
    """
    Generate a word embedding by averaging embeddings of individual words in the text.
    
    Args:
        text: Input text.
        embedding_dict: Preloaded word embeddings (e.g., GloVe or FastText).
        
    Returns:
        Averaged word embedding vector.
    """
    words = text.lower().split()
    vectors = [embedding_dict[word] for word in words if word in embedding_dict]
    if vectors:
        return torch.mean(torch.stack(vectors), dim=0).cpu().numpy()
    return np.zeros(300)  # Return zero vector if no words match

def create_attention_based_embedding(albert_emb, additional_emb):
    """
    Fuse ALBERT embedding with additional word embedding using attention-based fusion.
    
    Args:
        albert_emb: ALBERT embedding vector.
        additional_emb: Additional word embedding vector.
        
    Returns:
        Fused embedding vector.
    """
    if albert_emb.shape != additional_emb.shape:
        additional_emb = torch.nn.Linear(additional_emb.shape[0], albert_emb.shape[0]).to(albert_emb.device)(additional_emb)
    combined_emb = torch.cat([albert_emb.unsqueeze(0), additional_emb.unsqueeze(0)], dim=0)
    attention_weights = torch.nn.Parameter(torch.tensor([0.5, 0.5], device=albert_emb.device), requires_grad=True)
    attention_scores = F.softmax(attention_weights, dim=0)
    fused_embedding = attention_scores[0] * albert_emb + attention_scores[1] * additional_emb
    return fused_embedding

def create_combined_embedding(text, embedding_type=None, _glove_model=None, _fasttext_model=None):
    """
    Create a combined embedding for a given text, optionally incorporating additional embeddings.

    Args:
        text: Input text for which to create the embedding.
        embedding_type: Type of additional embedding to use ("glove" or "fasttext").
        _glove_model: Preloaded GloVe embeddings, if using GloVe.
        _fasttext_model: Preloaded FastText embeddings, if using FastText.

    Returns:
        A tuple containing the combined embedding and its size.
    """

    albert_emb = get_albert_embedding(text).flatten()

    if embedding_type == "glove":
        additional_emb = get_word_embedding(text, _glove_model)
    elif embedding_type == "fasttext":
        additional_emb = get_word_embedding(text, _fasttext_model)
    else:
        additional_emb = np.array([])

    albert_emb_tensor = torch.tensor(albert_emb, dtype=torch.float32).to(device)

    if additional_emb.size != 0:
        additional_emb_tensor = torch.tensor(additional_emb, dtype=torch.float32).to(device)
        
        # Ensure both embeddings have the same size by truncating or padding
        if additional_emb_tensor.size(0) > albert_emb_tensor.size(0):
            additional_emb_tensor = additional_emb_tensor[:albert_emb_tensor.size(0)]  # Truncate
        elif additional_emb_tensor.size(0) < albert_emb_tensor.size(0):
            padding_size = albert_emb_tensor.size(0) - additional_emb_tensor.size(0)
            additional_emb_tensor = F.pad(additional_emb_tensor, (0, padding_size))  # Pad with zeros

        combined_emb = torch.cat([albert_emb_tensor, additional_emb_tensor], dim=0)
    else:
        combined_emb = albert_emb_tensor

    return combined_emb.cpu().numpy(), combined_emb.size(0)

def train_and_save_model(X_train_tensor, y_train_tensor, y_train_quality_tensor, y_train_essay_type_tensor, 
                         y_train_content_tensor, y_train_organization_tensor, y_train_word_choice_tensor, 
                         y_train_sentence_fluency_tensor, y_train_conventions_tensor, y_train_language_tensor, 
                         y_train_prompt_adherence_tensor, y_train_narrativity_tensor, input_shape, save_dir, 
                         embedding_type=None, epochs=10, batch_size=8, learning_rate=1e-4):

    """
    Train a multitask model on essay data for regression and classification tasks, and save the trained model.

    This function trains a multitask neural network on a dataset of essay features, where the model predicts an 
    overall essay score (regression), quality category, essay type, and several attribute scores such as content, 
    organization, word choice, sentence fluency, and conventions. The function uses label smoothing for classification
    tasks to reduce overfitting, and task uncertainty weighting for loss balancing between different tasks.

    Parameters:
    ----------
    X_train_tensor : torch.Tensor
        The training input features as a PyTorch tensor.
    y_train_tensor : torch.Tensor
        Ground truth scores for the primary regression task.
    y_train_quality_tensor : torch.Tensor
        Ground truth labels for quality classification.
    y_train_essay_type_tensor : torch.Tensor
        Ground truth labels for essay type classification.
    y_train_content_tensor : torch.Tensor
        Ground truth labels for the content attribute.
    y_train_organization_tensor : torch.Tensor
        Ground truth labels for the organization attribute.
    y_train_word_choice_tensor : torch.Tensor
        Ground truth labels for the word choice attribute.
    y_train_sentence_fluency_tensor : torch.Tensor
        Ground truth labels for the sentence fluency attribute.
    y_train_conventions_tensor : torch.Tensor
        Ground truth labels for the conventions attribute.
    input_shape : int
        The shape (number of features) of the input to the model.
    save_dir : str
        The directory to save the trained model and embedding size file.
    epochs : int, optional, default=10
        The number of epochs to train the model.
    batch_size : int, optional, default=8
        The number of samples per batch for training.
    learning_rate : float, optional, default=1e-4
        The learning rate for the optimizer.

    Returns:
    -------
    str
        The file path of the saved model.
    
    Notes:
    ------
    - This function computes a combined loss across multiple tasks, including regression for score prediction,
      classification for quality and essay type, and additional regression tasks for specific attributes.
    - The model is saved at the end of training along with the embedding size to facilitate consistent embedding 
      processing during inference.
    """

    model = MultiTaskModel(input_shape).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    train_loader = DataLoader(TensorDataset(
            X_train_tensor, y_train_tensor, y_train_quality_tensor, y_train_essay_type_tensor, y_train_content_tensor, 
            y_train_organization_tensor, y_train_word_choice_tensor, y_train_sentence_fluency_tensor,
            y_train_conventions_tensor, y_train_language_tensor, y_train_prompt_adherence_tensor, y_train_narrativity_tensor), 
            batch_size=batch_size, shuffle=True
        )


    # Initialize Label Smoothing loss
    label_smoothing_loss = LabelSmoothingCrossEntropy(smoothing=0.1)
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0

        for X_batch, y_score_batch, y_quality_batch, y_essay_type_batch, y_content_batch, y_organization_batch, \
            y_word_choice_batch, y_sentence_fluency_batch, y_conventions_batch, y_language_batch, \
            y_prompt_adherence_batch, y_narrativity_batch in train_loader:

            # Move data to device
            X_batch, y_score_batch, y_quality_batch, y_essay_type_batch = X_batch.to(device), y_score_batch.to(device), y_quality_batch.to(device), y_essay_type_batch.to(device)
            y_content_batch, y_organization_batch, y_word_choice_batch = y_content_batch.to(device), y_organization_batch.to(device), y_word_choice_batch.to(device)
            y_sentence_fluency_batch, y_conventions_batch = y_sentence_fluency_batch.to(device), y_conventions_batch.to(device)
            y_language_batch, y_prompt_adherence_batch, y_narrativity_batch = y_language_batch.to(device), y_prompt_adherence_batch.to(device), y_narrativity_batch.to(device)
            
            optimizer.zero_grad()
            
            # Unpack all 13 outputs from the model
            pred_score, pred_quality, pred_essay_type, pred_content, pred_organization, pred_word_choice, \
            pred_sentence_fluency, pred_conventions, pred_language, pred_prompt_adherence, pred_narrativity = model(X_batch)

            # Calculate the losses
            mse_loss = nn.MSELoss()(pred_score, y_score_batch).mean()  # Ensure it's a scalar
            cross_entropy_loss_quality = label_smoothing_loss(pred_quality, y_quality_batch).mean()  # Ensure it's a scalar
            cross_entropy_loss_essay_type = label_smoothing_loss(pred_essay_type, y_essay_type_batch).mean()  # Ensure it's a scalar

            # MSE loss for the additional attributes
            mse_loss_content = nn.MSELoss()(pred_content, y_content_batch)
            mse_loss_organization = nn.MSELoss()(pred_organization, y_organization_batch)
            mse_loss_word_choice = nn.MSELoss()(pred_word_choice, y_word_choice_batch)
            mse_loss_sentence_fluency = nn.MSELoss()(pred_sentence_fluency, y_sentence_fluency_batch)
            mse_loss_conventions = nn.MSELoss()(pred_conventions, y_conventions_batch)
            mse_loss_language = nn.MSELoss()(pred_language, y_language_batch)
            mse_loss_prompt_adherence = nn.MSELoss()(pred_prompt_adherence, y_prompt_adherence_batch)
            mse_loss_narrativity = nn.MSELoss()(pred_narrativity, y_narrativity_batch)

            # Compute uncertainty loss
            uncertainty_loss = model.compute_uncertainty_loss(mse_loss, cross_entropy_loss_quality).mean()  # Ensure it's a scalar

            # Total loss is a combination of all these
            total_loss = (mse_loss + cross_entropy_loss_quality + cross_entropy_loss_essay_type + uncertainty_loss) / 4  # Average loss
            total_loss += mse_loss_content + mse_loss_organization + mse_loss_word_choice + \
                mse_loss_sentence_fluency + mse_loss_conventions + mse_loss_language + \
                mse_loss_prompt_adherence + mse_loss_narrativity

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)  # Adjust max_norm if necessary
            optimizer.step()
            epoch_loss += total_loss.item()

        print(f"Epoch {epoch + 1}/{epochs}, Total Epoch Loss: {epoch_loss / len(train_loader):.4f}")

    model_filename = f"albert_model_{embedding_type or 'albert'}.pth"
    embedding_size_filename = f"albert_embedding_size_{embedding_type or 'albert'}.npy"
    torch.save(model.state_dict(), os.path.join(save_dir, model_filename))
    np.save(os.path.join(save_dir, embedding_size_filename), input_shape)
    
    # Return only the model file path
    return os.path.join(save_dir, model_filename)

def evaluate_model(model_path, X_test_tensor, y_test, y_test_quality, y_test_essay_type, 
                   y_test_content, y_test_organization, y_test_word_choice, y_test_sentence_fluency,
                   y_test_conventions, y_test_language, y_test_prompt_adherence, y_test_narrativity, save_dir, model_name):

    """
    Evaluate a multitask model on test data and visualize results, including confusion matrices and kappa heatmaps.

    This function loads a pretrained multitask model, computes regression and classification metrics on test data, 
    and generates visualizations to assess model performance. It calculates metrics for overall essay score, quality, 
    and essay type classifications, as well as kappa scores for content, organization, word choice, sentence fluency, 
    and conventions attributes.

    Parameters:
    ----------
    model_path : str
        Path to the saved model file.
    X_test_tensor : torch.Tensor
        Test features as a PyTorch tensor.
    y_test : torch.Tensor or np.ndarray
        Ground truth scores for the primary regression task.
    y_test_quality : torch.Tensor
        Ground truth labels for quality classification.
    y_test_essay_type : torch.Tensor
        Ground truth labels for essay type classification.
    y_test_content : torch.Tensor
        Ground truth labels for the content attribute.
    y_test_organization : torch.Tensor
        Ground truth labels for the organization attribute.
    y_test_word_choice : torch.Tensor
        Ground truth labels for the word choice attribute.
    y_test_sentence_fluency : torch.Tensor
        Ground truth labels for the sentence fluency attribute.
    y_test_conventions : torch.Tensor
        Ground truth labels for the conventions attribute.
    save_dir : str
        Directory where model files are stored (not directly used but may be needed for loading auxiliary files).
    model_name : str
        Name of the model being evaluated, used for labeling in visualizations.

    Returns:
    -------
    tuple
        A tuple containing various evaluation metrics:
        - MSE for score
        - Accuracy, F1 score, and Kappa for quality classification
        - Accuracy, F1 score, and Kappa for essay type classification
        - Kappa scores for content, organization, word choice, sentence fluency, and conventions attributes.
    
    Notes:
    ------
    - Confusion matrices are generated for both quality and essay type classifications.
    - Kappa scores for each attribute are visualized in a heatmap.
    """

    # Load the model and move it to the appropriate device
    model = MultiTaskModel(X_test_tensor.shape[1]).to(device)  # Move model to device
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Move test tensors to the correct device
    X_test_tensor = X_test_tensor.to(device)
    y_test_quality = y_test_quality.to(device)
    y_test_essay_type = y_test_essay_type.to(device)
    y_test_content = y_test_content.to(device)
    y_test_organization = y_test_organization.to(device)
    y_test_word_choice = y_test_word_choice.to(device)
    y_test_sentence_fluency = y_test_sentence_fluency.to(device)
    y_test_conventions = y_test_conventions.to(device)
    y_test_language = y_test_language.to(device)
    y_test_prompt_adherence = y_test_prompt_adherence.to(device)
    y_test_narrativity = y_test_narrativity.to(device)

    with torch.no_grad():
        # Get model predictions (all outputs)
        pred_scores, pred_qualities, pred_essay_types, pred_content, pred_organization, pred_word_choice, \
        pred_sentence_fluency, pred_conventions, pred_language, pred_prompt_adherence, pred_narrativity = model(X_test_tensor)

        # Ensure y_test is a tensor and move to CPU if it's a numpy array
        if isinstance(y_test, np.ndarray):
            y_test = torch.tensor(y_test, dtype=torch.float32).to(device)

        # Regression metrics: Mean Squared Error (MSE) for score
        mse = mean_squared_error(y_test.cpu().numpy(), pred_scores.cpu().numpy().squeeze())
        
        # Classification metrics for quality
        y_pred_qualities = pred_qualities.argmax(dim=1).cpu().numpy()
        f1_quality = f1_score(y_test_quality.cpu().numpy(), y_pred_qualities, average='weighted')
        accuracy_quality = accuracy_score(y_test_quality.cpu().numpy(), y_pred_qualities)
        kappa_quality = cohen_kappa_score(y_test_quality.cpu().numpy(), y_pred_qualities, weights='quadratic')  # Quadratic Kappa for quality
        
        # Classification metrics for essay type
        y_pred_essay_types = pred_essay_types.argmax(dim=1).cpu().numpy()
        f1_essay_type = f1_score(y_test_essay_type.cpu().numpy(), y_pred_essay_types, average='weighted')
        accuracy_essay_type = accuracy_score(y_test_essay_type.cpu().numpy(), y_pred_essay_types)
        kappa_essay_type = cohen_kappa_score(y_test_essay_type.cpu().numpy(), y_pred_essay_types, weights='quadratic')  # Quadratic Kappa for essay type

        # Kappa scores for each attribute
        y_pred_content = pred_essay_types.argmax(dim=1).cpu().numpy()
        kappa_content = cohen_kappa_score(y_test_content.cpu().numpy(), y_pred_content, weights='quadratic')

        y_pred_organization = pred_essay_types.argmax(dim=1).cpu().numpy()
        kappa_organization = cohen_kappa_score(y_test_organization.cpu().numpy(), y_pred_organization, weights='quadratic')

        y_pred_word_choice = pred_essay_types.argmax(dim=1).cpu().numpy()
        kappa_word_choice = cohen_kappa_score(y_test_word_choice.cpu().numpy(), y_pred_word_choice, weights='quadratic')

        y_pred_sentence_fluency = pred_essay_types.argmax(dim=1).cpu().numpy()
        kappa_sentence_fluency = cohen_kappa_score(y_test_sentence_fluency.cpu().numpy(), y_pred_sentence_fluency, weights='quadratic')

        y_pred_conventions = pred_essay_types.argmax(dim=1).cpu().numpy()
        kappa_conventions = cohen_kappa_score(y_test_conventions.cpu().numpy(), y_pred_conventions, weights='quadratic')

        y_pred_language = pred_essay_types.argmax(dim=1).cpu().numpy()
        kappa_language = cohen_kappa_score(y_test_language.cpu().numpy(), y_pred_language, weights='quadratic')
        
        y_pred_prompt_adherence = pred_essay_types.argmax(dim=1).cpu().numpy()
        kappa_prompt_adherence = cohen_kappa_score(y_test_prompt_adherence.cpu().numpy(), y_pred_prompt_adherence, weights='quadratic')

        y_pred_narrativity = pred_essay_types.argmax(dim=1).cpu().numpy()
        kappa_narrativity = cohen_kappa_score(y_test_narrativity.cpu().numpy(), y_pred_narrativity, weights='quadratic')

        plot_confusion_matrices(y_test_quality.cpu().numpy(), y_pred_qualities, y_test_essay_type.cpu().numpy(), y_pred_essay_types)

        attribute_kappa_scores = [kappa_content, kappa_organization, kappa_word_choice, kappa_sentence_fluency, kappa_conventions, kappa_language, kappa_prompt_adherence, kappa_narrativity]
        plot_kappa_heatmap([attribute_kappa_scores], model_names=[model_name], attribute_names=['Content', 'Organization', 'Word Choice', 'Sentence Fluency', 'Conventions', 'Language', 'Prompt Adherence', 'Narrativity'])


    # Print out the evaluation results
    print(f"Evaluation Results: \nMSE for Score: {mse}")
    print(f"Quality Classification Accuracy: {accuracy_quality:.5f}")
    print(f"Quality Classification F1 Score: {f1_quality:.5f}")
    print(f"Quality Classification Quadratic Kappa: {kappa_quality:.5f}")
    print(f"Essay Type Classification Accuracy: {accuracy_essay_type:.5f}")
    print(f"Essay Type Classification F1 Score: {f1_essay_type:.5f}")
    print(f"Essay Type Classification Quadratic Kappa: {kappa_essay_type:.5f}")

    # Print out Kappa scores for each attribute
    print(f"Kappa for Content: {kappa_content:.5f}")
    print(f"Kappa for Organization: {kappa_organization:.5f}")
    print(f"Kappa for Word Choice: {kappa_word_choice:.5f}")
    print(f"Kappa for Sentence Fluency: {kappa_sentence_fluency:.5f}")
    print(f"Kappa for Conventions: {kappa_conventions:.5f}")
    print(f"Kappa for Language: {kappa_language:.5f}")
    print(f"Kappa for Prompt Adherence: {kappa_prompt_adherence:.5f}")
    print(f"Kappa for Narrativity: {kappa_narrativity:.5f}")

    return (
        mse, accuracy_quality, f1_quality, kappa_quality, accuracy_essay_type, f1_essay_type, kappa_essay_type,
        kappa_content, kappa_organization, kappa_word_choice, kappa_sentence_fluency, kappa_conventions,
        kappa_language, kappa_prompt_adherence, kappa_narrativity
    )


def testContent(content, embedding_type=None, SAVE_DIR=None, glove_model=None, fasttext_model=None, min_score=0, max_score=100, attribute_ranges=None):
    """
    Generate predictions for a given essay content using the specified model and embeddings.

    Args:
        content (str): The text of the essay to evaluate.
        embedding_type (str, optional): The type of additional embedding to use ("glove" or "fasttext").
        SAVE_DIR (str, optional): Directory where model files are stored.
        glove_model (dict, optional): Preloaded GloVe embeddings if using GloVe.
        fasttext_model (dict, optional): Preloaded FastText embeddings if using FastText.
        min_score (int, optional): Minimum score range for normalizing the predicted score.
        max_score (int, optional): Maximum score range for normalizing the predicted score.
        attribute_ranges (dict, optional): Dictionary containing min and max ranges for each attribute in the dataset.

    Returns:
        tuple: A tuple containing:
            - formatted_score (float): The normalized and formatted score between 0 and 100.
            - quality_label (str): The quality level ("Low", "Medium", or "High").
            - essay_type (str): The type of essay ("Argumentative", "Dependent", or "Narrative").
            - content_score, organization_score, word_choice_score, sentence_fluency_score, conventions_score (int):
              Normalized scores for each of these specific attributes.
    """
    
    if attribute_ranges is None:
        raise ValueError("attribute_ranges must be provided.")
    
    # Ensure content is a valid string or convert it if necessary
    if content is None:
        raise ValueError("Content cannot be None.")
    elif isinstance(content, list):
        content = " ".join(content)  # Join list elements into a single string
    elif not isinstance(content, str):
        content = str(content)  # Convert any other type to a string
    
    # Generate the combined embedding
    embedding, actual_embedding_size = create_combined_embedding(
        content,
        embedding_type=embedding_type,
        _glove_model=glove_model if embedding_type == "glove" else None,
        _fasttext_model=fasttext_model if embedding_type == "fasttext" else None
    )

    embedding_tensor = torch.tensor(embedding, dtype=torch.float32).to(device).unsqueeze(0)

    # Load model files
    embedding_size_filename = f"albert_embedding_size_{embedding_type or 'albert'}.npy"
    model_filename = f"albert_model_{embedding_type or 'albert'}.pth"
    
    # Load the expected embedding size and model
    embedding_size_path = os.path.join(SAVE_DIR, embedding_size_filename)
    expected_embedding_size = int(np.load(embedding_size_path))

    # Initialize model and load weights
    model = MultiTaskModel(expected_embedding_size).to(device)
    model_path = os.path.join(SAVE_DIR, model_filename)
    state_dict = torch.load(model_path, map_location=device)
    filtered_state_dict = {k: v for k, v in state_dict.items() if k in model.state_dict()}
    model.load_state_dict(filtered_state_dict, strict=False)
    model.eval()

    # Adjust embedding size if necessary
    embedding_resized = embedding_tensor[:, :expected_embedding_size]

    # Make predictions
    with torch.no_grad():
        pred_score, pred_quality, pred_essay_type, pred_content, pred_organization, pred_word_choice, \
        pred_sentence_fluency, pred_conventions, pred_language, pred_prompt_adherence, pred_narrativity = model(embedding_resized)
        
        # Retrieve raw predictions for all attributes
        raw_score = pred_score.cpu().item()
        quality_label_idx = pred_quality.argmax(dim=1).cpu().item()
        essay_type_idx = pred_essay_type.argmax(dim=1).cpu().item()

        # Normalize and round up the attribute scores based on their ranges
        content_score = normalize_and_round_up(pred_content.cpu().item(), *attribute_ranges['content'])
        organization_score = normalize_and_round_up(pred_organization.cpu().item(), *attribute_ranges['organization'])
        word_choice_score = normalize_and_round_up(pred_word_choice.cpu().item(), *attribute_ranges['word_choice'])
        sentence_fluency_score = normalize_and_round_up(pred_sentence_fluency.cpu().item(), *attribute_ranges['sentence_fluency'])
        conventions_score = normalize_and_round_up(pred_conventions.cpu().item(), *attribute_ranges['conventions'])
        language_score = normalize_and_round_up(pred_language.cpu().item(), *attribute_ranges['language'])
        prompt_adherence_score = normalize_and_round_up(pred_prompt_adherence.cpu().item(), *attribute_ranges['prompt_adherence'])
        narrativity_score = normalize_and_round_up(pred_narrativity.cpu().item(), *attribute_ranges['narrativity'])

    # Normalize the overall score to a 0-100 range based on min and max score from training
    normalized_score = (raw_score - min_score) / (max_score - min_score) * 100
    normalized_score = max(0, min(100, normalized_score))

    # Map quality label index to label
    quality_mapping = {0: "Low", 1: "Medium", 2: "High"}
    quality_label = quality_mapping[quality_label_idx]

    # Map essay type index to type
    essay_type_mapping = {0: "Argumentative", 1: "Dependent"}
    essay_type = essay_type_mapping[essay_type_idx]

    # Return all predictions
    formatted_score = round(normalized_score, 5)
    return formatted_score, quality_label, essay_type, content_score, organization_score, word_choice_score, \
           sentence_fluency_score, conventions_score, language_score, prompt_adherence_score, narrativity_score
           
def generate_feedback(content, score, quality_level):
    """
    Generate feedback based on essay content, score, and quality level.

    Args:
        content: Essay content.
        score: Predicted score.
        quality_level: Predicted quality level.

    Returns:
        Generated feedback text.
    """

    quality_text = {0: "low", 1: "medium", 2: "high"}.get(quality_level, "unknown")
    
    # Construct the prompt to ensure structured and concise feedback
    prompt = f"""The following is an essay content with a predicted score of {score} and a quality level of {quality_text}:
    
    Essay:
    {content}
    
    Based on this score and quality level, provide a structured feedback response in the following format:
    - Strengths: List 3 brief points about what the essay does well.
    - Areas of Improvement:
        - Content: Suggest one sentence on how the content could be improved.
        - Grammar: Suggest one sentence on how the grammar could be improved.
        
    Please keep the response concise."""

    try:
        # Make the request to Groq's chat completion API
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a helpful assistant that provides concise and structured feedback for essays."},
                {"role": "user", "content": prompt}
            ],
            model="gemma-7b-it",
            temperature=0.5,
            max_tokens=150
        )

        # Extract and return structured feedback from the response
        feedback = chat_completion.choices[0].message.content.strip()
        return feedback
    except Exception as e:
        return f"Error generating feedback: {e}"

def get_attribute_range(df, attribute_name):
    """Get min and max range for a given attribute in the dataset."""
    min_val = df[attribute_name].min()
    max_val = df[attribute_name].max()
    return min_val, max_val

def normalize_and_round_up(attribute_value, min_value, max_value):
    """Normalize and round up the attribute value to the specified range."""
    # Check for NaN and replace with 0
    if np.isnan(attribute_value):
        attribute_value = 0

    # Handle case where min_value and max_value are the same (e.g., range is 0 to 0)
    if min_value == max_value:
        return 0

    # Normalize the value to the range [0, 1] first
    normalized_value = (attribute_value - min_value) / (max_value - min_value)
    
    # Scale to the desired range and round it up
    scaled_value = normalized_value * (max_value - min_value) + min_value
    
    # Use math.ceil to round up
    return math.ceil(scaled_value)

def display_selected_attributes(essay_type, attributes):
    """
    Display only relevant attributes based on the essay type.

    Args:
        essay_type: Type of essay (e.g., Argumentative, Dependent, Narrative).
        attributes: Dictionary of attribute scores.
    """    
    # Define the attribute mappings for each essay type
    attribute_mapping = {
        "Argumentative": ['content', 'organization', 'word_choice', 'sentence_fluency', 'conventions'],
        "Dependent": ['content', 'language', 'prompt_adherence', 'narrativity'],
    }
    
    # Get the relevant attributes based on the essay type
    relevant_attributes = attribute_mapping.get(essay_type, [])
    
    # Display only the relevant attributes
    print(f"\nEssay Type: {essay_type}")
    for attr in relevant_attributes:
        print(f"{attr.capitalize().replace('_', ' ')}: {attributes[attr]}")

def plot_confusion_matrices(y_test_quality, y_pred_qualities, y_test_essay_type, y_pred_essay_types):
    """
    Plot confusion matrices for quality and essay type classifications.

    This function creates side-by-side confusion matrix plots to visualize 
    the model's performance on quality and essay type classification tasks.

    Parameters:
    ----------
    y_test_quality : array-like
        True labels for the quality classification task.
    y_pred_qualities : array-like
        Predicted labels for the quality classification task.
    y_test_essay_type : array-like
        True labels for the essay type classification task.
    y_pred_essay_types : array-like
        Predicted labels for the essay type classification task.
    """
    # Plot confusion matrix for quality classification
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot confusion matrix for quality classification
    cm_quality = confusion_matrix(y_test_quality, y_pred_qualities)
    ConfusionMatrixDisplay(cm_quality).plot(ax=ax[0], cmap='Blues', values_format='d')
    ax[0].set_title("Confusion Matrix for Quality Classification")

    # Plot confusion matrix for essay type classification
    cm_essay_type = confusion_matrix(y_test_essay_type, y_pred_essay_types)
    ConfusionMatrixDisplay(cm_essay_type).plot(ax=ax[1], cmap='Greens', values_format='d')
    ax[1].set_title("Confusion Matrix for Essay Type Classification")
    
    # Adjust layout for better viewing
    plt.tight_layout()
    plt.show()

def plot_kappa_heatmap(kappa_scores, model_names, attribute_names):
    """
    Plot a heatmap of Kappa scores across models and attributes.

    This function visualizes the Kappa scores for each model and attribute 
    to assess agreement levels for content, organization, word choice, 
    sentence fluency, and conventions.

    Parameters:
    ----------
    kappa_scores : list of lists or 2D array
        Kappa scores for each model and attribute. Each inner list contains scores for a specific model.
    model_names : list of str
        Names of the models, used as row labels in the heatmap.
    attribute_names : list of str
        Names of the attributes, used as column labels in the heatmap.
    """
    
    # Ensure kappa_scores is a DataFrame-compatible structure
    kappa_df = pd.DataFrame(kappa_scores, index=model_names, columns=attribute_names)
    
    # Create a heatmap to visualize Kappa scores across attributes and models
    plt.figure(figsize=(10, 6))
    sns.heatmap(kappa_df, annot=True, cmap="coolwarm", cbar_kws={'label': 'Kappa Score'}, fmt=".5f")
    plt.title("Kappa Scores for Each Attribute Across Models")
    plt.xlabel("Attribute")
    plt.ylabel("Model Variant")
    plt.show()

def plot_training_history(train_losses, val_losses):
    """
    Plot training and validation loss over epochs to evaluate model performance.

    This function visualizes the loss for training and validation sets over 
    epochs, helping to assess whether the model may be underfitting or overfitting.

    Parameters:
    ----------
    train_losses : list
        Training loss values for each epoch.
    val_losses : list
        Validation loss values for each epoch.
    """

    # Plot the training and validation loss curves
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Over Epochs')
    plt.legend()
    plt.show()