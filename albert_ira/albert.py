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
import torch.nn.functional as F
import requests
import os
from groq import Groq
import math

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# Constants
MODEL_NAME = "albert-base-v2"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load ALBERT tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
albert_model = AutoModel.from_pretrained(MODEL_NAME).to(device)

# Initialize Groq client
client = Groq(api_key="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")


# Define a regression model for predicting scores
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
    """A multitask model for predicting both regression (score) and classification (quality)."""
    def __init__(self, input_shape):
        super(MultiTaskModel, self).__init__()
        # Shared layers
        self.fc1 = nn.Linear(input_shape, 256)
        self.bn1 = nn.BatchNorm1d(256)  # Batch normalization
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(0.5)
        
        self.regression_head = nn.Linear(128, 1)
        self.classification_head = nn.Linear(128, 3)  # 3 classes: low, medium, high
        self.essay_type_head = nn.Linear(128, 3)  # 3 types: argumentative, dependent, narrative

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
        self.style_head = nn.Linear(128, 1)  # Style
        self.voice_head = nn.Linear(128, 1)  # Voice
     
    def forward(self, x):
        x = torch.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        x = torch.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        
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
        style_output = self.style_head(x)
        voice_output = self.voice_head(x)

        return (score_output, quality_output, essay_type_output,
                content_output, organization_output, word_choice_output,
                sentence_fluency_output, conventions_output, language_output,
                prompt_adherence_output, narrativity_output, style_output, voice_output)

    def compute_uncertainty_loss(self, loss_score, loss_quality):
        precision1 = torch.exp(-self.task_uncertainty[0])
        precision2 = torch.exp(-self.task_uncertainty[1])
        
        loss = (precision1 * loss_score + self.task_uncertainty[0]) + \
               (precision2 * loss_quality + self.task_uncertainty[1])
        
        return loss
        
    def compute_loss(self, pred_score, pred_quality, pred_essay_type,
                     pred_content, pred_organization, pred_word_choice,
                     pred_sentence_fluency, pred_conventions, pred_language,
                     pred_prompt_adherence, pred_narrativity, pred_style, pred_voice,
                     y_score, y_quality, y_essay_type,
                     y_content, y_organization, y_word_choice,
                     y_sentence_fluency, y_conventions, y_language,
                     y_prompt_adherence, y_narrativity, y_style, y_voice):
        
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
        mse_loss_style = nn.MSELoss()(pred_style, y_style)
        mse_loss_voice = nn.MSELoss()(pred_voice, y_voice)

        # Combine all the losses
        total_loss = mse_loss + cross_entropy_loss_quality + cross_entropy_loss_essay_type + \
                     mse_loss_content + mse_loss_organization + mse_loss_word_choice + \
                     mse_loss_sentence_fluency + mse_loss_conventions + mse_loss_language + \
                     mse_loss_prompt_adherence + mse_loss_narrativity + mse_loss_style + mse_loss_voice
        
        return total_loss


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing

    def forward(self, pred, target):
        log_probs = F.log_softmax(pred, dim=-1)
        nll_loss = -log_probs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -log_probs.mean(dim=-1)
        return (1 - self.smoothing) * nll_loss + self.smoothing * smooth_loss

def load_glove_model(glove_file_path):
    """Load GloVe embeddings from file into a dictionary."""
    embedding_dict = {}
    with open(glove_file_path, 'r', encoding="utf-8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = torch.tensor(np.asarray(values[1:], dtype='float32'))
            embedding_dict[word] = vector.to(device)  # Move to device if necessary
    return embedding_dict

def load_fasttext_model(fasttext_file_path):
    """Load FastText embeddings from file into a dictionary."""
    model = KeyedVectors.load_word2vec_format(fasttext_file_path, binary=False)
    return {word: torch.tensor(model[word]).to(device) for word in model.index_to_key}

def get_albert_embedding(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=256).to(device)
    with torch.no_grad():
        outputs = albert_model(**inputs)
    return outputs.last_hidden_state[:, 0, :].cpu().numpy()

def get_word_embedding(text, embedding_dict):
    words = text.lower().split()
    vectors = [embedding_dict[word] for word in words if word in embedding_dict]
    if vectors:
        return torch.mean(torch.stack(vectors), dim=0).cpu().numpy()
    return np.zeros(300)

# Create attention-based embedding fusion
def create_attention_based_embedding(albert_emb, additional_emb):
    if albert_emb.shape != additional_emb.shape:
        additional_emb = torch.nn.Linear(additional_emb.shape[0], albert_emb.shape[0]).to(albert_emb.device)(additional_emb)
    combined_emb = torch.cat([albert_emb.unsqueeze(0), additional_emb.unsqueeze(0)], dim=0)
    attention_weights = torch.nn.Parameter(torch.tensor([0.5, 0.5], device=albert_emb.device), requires_grad=True)
    attention_scores = F.softmax(attention_weights, dim=0)
    fused_embedding = attention_scores[0] * albert_emb + attention_scores[1] * additional_emb
    return fused_embedding

def create_combined_embedding(text, embedding_type=None, _glove_model=None, _fasttext_model=None):
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

def train_and_save_model(X_train_tensor, y_train_tensor, y_train_quality_tensor, y_train_essay_type_tensor, input_shape, save_dir, epochs=10, batch_size=8, learning_rate=1e-4):
    model = MultiTaskModel(input_shape).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor, y_train_quality_tensor, y_train_essay_type_tensor, y_train_content_tensor, 
                                             y_train_organization_tensor, y_train_word_choice_tensor, y_train_sentence_fluency_tensor,
                                             y_train_conventions_tensor, y_train_language_tensor, y_train_prompt_adherence_tensor,
                                             y_train_narrativity_tensor, y_train_style_tensor, y_train_voice_tensor), 
                              batch_size=batch_size, shuffle=True)

    # Initialize Label Smoothing loss
    label_smoothing_loss = LabelSmoothingCrossEntropy(smoothing=0.1)
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for X_batch, y_score_batch, y_quality_batch, y_essay_type_batch, y_content_batch, y_organization_batch, \
        y_word_choice_batch, y_sentence_fluency_batch, y_conventions_batch, y_language_batch, \
        y_prompt_adherence_batch, y_narrativity_batch, y_style_batch, y_voice_batch in train_loader:
            
            # Move data to device
            X_batch, y_score_batch, y_quality_batch, y_essay_type_batch = X_batch.to(device), y_score_batch.to(device), y_quality_batch.to(device), y_essay_type_batch.to(device)
            y_content_batch, y_organization_batch, y_word_choice_batch = y_content_batch.to(device), y_organization_batch.to(device), y_word_choice_batch.to(device)
            y_sentence_fluency_batch, y_conventions_batch, y_language_batch = y_sentence_fluency_batch.to(device), y_conventions_batch.to(device), y_language_batch.to(device)
            y_prompt_adherence_batch, y_narrativity_batch, y_style_batch, y_voice_batch = y_prompt_adherence_batch.to(device), y_narrativity_batch.to(device), y_style_batch.to(device), y_voice_batch.to(device)
            
            optimizer.zero_grad()
            
            # Unpack all 13 outputs from the model
            pred_score, pred_quality, pred_essay_type, pred_content, pred_organization, pred_word_choice, \
            pred_sentence_fluency, pred_conventions, pred_language, pred_prompt_adherence, pred_narrativity, \
            pred_style, pred_voice = model(X_batch)

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
            mse_loss_style = nn.MSELoss()(pred_style, y_style_batch)
            mse_loss_voice = nn.MSELoss()(pred_voice, y_voice_batch)

            # Compute uncertainty loss
            uncertainty_loss = model.compute_uncertainty_loss(mse_loss, cross_entropy_loss_quality).mean()  # Ensure it's a scalar

            # Total loss is a combination of all these
            total_loss = (mse_loss + cross_entropy_loss_quality + cross_entropy_loss_essay_type + uncertainty_loss) / 4  # Average loss
            total_loss += mse_loss_content + mse_loss_organization + mse_loss_word_choice + \
                          mse_loss_sentence_fluency + mse_loss_conventions + mse_loss_language + \
                          mse_loss_prompt_adherence + mse_loss_narrativity + mse_loss_style + mse_loss_voice

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)  # Adjust max_norm if necessary
            optimizer.step()
            epoch_loss += total_loss.item()

        print(f"Epoch {epoch + 1}/{epochs}, Total Epoch Loss: {epoch_loss / len(train_loader):.4f}")

    model_filename = f"albert2_model_{embedding_type or 'albert'}.pth"
    embedding_size_filename = f"albert2_embedding_size_{embedding_type or 'albert'}.npy"
    torch.save(model.state_dict(), os.path.join(save_dir, model_filename))
    np.save(os.path.join(save_dir, embedding_size_filename), input_shape)
    
    # Return only the model file path
    return os.path.join(save_dir, model_filename)

def evaluate_model(model_path, X_test_tensor, y_test, y_test_quality, y_test_essay_type, 
                   y_test_content, y_test_organization, y_test_word_choice, y_test_sentence_fluency,
                   y_test_conventions, y_test_language, y_test_prompt_adherence, y_test_narrativity,
                   y_test_style, y_test_voice, save_dir):
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
    y_test_style = y_test_style.to(device)
    y_test_voice = y_test_voice.to(device)

    with torch.no_grad():
        # Get model predictions (all outputs)
        pred_scores, pred_qualities, pred_essay_types, pred_content, pred_organization, pred_word_choice, \
        pred_sentence_fluency, pred_conventions, pred_language, pred_prompt_adherence, pred_narrativity, \
        pred_style, pred_voice = model(X_test_tensor)
        
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
        
        # Regression metrics for the additional attributes (content, organization, word_choice, etc.)
        mse_content = mean_squared_error(y_test_content.cpu().numpy(), pred_content.cpu().numpy())
        mse_organization = mean_squared_error(y_test_organization.cpu().numpy(), pred_organization.cpu().numpy())
        mse_word_choice = mean_squared_error(y_test_word_choice.cpu().numpy(), pred_word_choice.cpu().numpy())
        mse_sentence_fluency = mean_squared_error(y_test_sentence_fluency.cpu().numpy(), pred_sentence_fluency.cpu().numpy())
        mse_conventions = mean_squared_error(y_test_conventions.cpu().numpy(), pred_conventions.cpu().numpy())
        mse_language = mean_squared_error(y_test_language.cpu().numpy(), pred_language.cpu().numpy())
        mse_prompt_adherence = mean_squared_error(y_test_prompt_adherence.cpu().numpy(), pred_prompt_adherence.cpu().numpy())
        mse_narrativity = mean_squared_error(y_test_narrativity.cpu().numpy(), pred_narrativity.cpu().numpy())
        mse_style = mean_squared_error(y_test_style.cpu().numpy(), pred_style.cpu().numpy())
        mse_voice = mean_squared_error(y_test_voice.cpu().numpy(), pred_voice.cpu().numpy())

    # Print out the evaluation results
    print(f"Evaluation Results: \nMSE for Score: {mse}")
    print(f"Quality Classification Accuracy: {accuracy_quality:.5f}")
    print(f"Quality Classification F1 Score: {f1_quality:.5f}")
    print(f"Quality Classification Quadratic Kappa: {kappa_quality:.5f}")
    print(f"Essay Type Classification Accuracy: {accuracy_essay_type:.5f}")
    print(f"Essay Type Classification F1 Score: {f1_essay_type:.5f}")
    print(f"Essay Type Classification Quadratic Kappa: {kappa_essay_type:.5f}")
    
    # Print out the MSE for each of the additional attributes
    print(f"MSE for Content: {mse_content:.5f}")
    print(f"MSE for Organization: {mse_organization:.5f}")
    print(f"MSE for Word Choice: {mse_word_choice:.5f}")
    print(f"MSE for Sentence Fluency: {mse_sentence_fluency:.5f}")
    print(f"MSE for Conventions: {mse_conventions:.5f}")
    print(f"MSE for Language: {mse_language:.5f}")
    print(f"MSE for Prompt Adherence: {mse_prompt_adherence:.5f}")
    print(f"MSE for Narrativity: {mse_narrativity:.5f}")
    print(f"MSE for Style: {mse_style:.5f}")
    print(f"MSE for Voice: {mse_voice:.5f}")

    return mse, accuracy_quality, f1_quality, kappa_quality, accuracy_essay_type, f1_essay_type, kappa_essay_type, \
           mse_content, mse_organization, mse_word_choice, mse_sentence_fluency, mse_conventions, mse_language, \
           mse_prompt_adherence, mse_narrativity, mse_style, mse_voice

def testContent(content, embedding_type=None, SAVE_DIR=None, glove_model=None, fasttext_model=None, min_score=0, max_score=100, attribute_ranges=None):
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
    embedding_size_filename = f"albert2_embedding_size_{embedding_type or 'albert'}.npy"
    model_filename = f"albert2_model_{embedding_type or 'albert'}.pth"
    
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
        pred_sentence_fluency, pred_conventions, pred_language, pred_prompt_adherence, pred_narrativity, \
        pred_style, pred_voice = model(embedding_resized)
        
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
        style_score = normalize_and_round_up(pred_style.cpu().item(), *attribute_ranges['style'])
        voice_score = normalize_and_round_up(pred_voice.cpu().item(), *attribute_ranges['voice'])

    # Normalize the overall score to a 0-100 range based on min and max score from training
    normalized_score = (raw_score - min_score) / (max_score - min_score) * 100
    normalized_score = max(0, min(100, normalized_score))

    # Map quality label index to label
    quality_mapping = {0: "Low", 1: "Medium", 2: "High"}
    quality_label = quality_mapping[quality_label_idx]

    # Map essay type index to type
    essay_type_mapping = {0: "Argumentative", 1: "Dependent", 2: "Narrative"}
    essay_type = essay_type_mapping[essay_type_idx]

    # Return all predictions
    formatted_score = round(normalized_score, 5)
    return formatted_score, quality_label, essay_type, content_score, organization_score, word_choice_score, \
           sentence_fluency_score, conventions_score, language_score, prompt_adherence_score, narrativity_score, \
           style_score, voice_score

def generate_feedback(content, score, quality_level):
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
    """Get the min and max range for a given attribute."""
    min_val = df[attribute_name].min()
    max_val = df[attribute_name].max()
    return min_val, max_val

def normalize_and_round_up(attribute_value, min_value, max_value):
    """Normalize the attribute value to the specified range [min_value, max_value] and round it up."""
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
    """Display only the attributes relevant to the specified essay type."""
    # Define the attribute mappings for each essay type
    attribute_mapping = {
        "Argumentative": ['content', 'organization', 'word_choice', 'sentence_fluency', 'conventions'],
        "Dependent": ['content', 'prompt_adherence', 'language', 'narrativity'],
        "Narrative": ['content', 'organization', 'style', 'conventions', 'voice', 'word_choice', 'sentence_fluency']
    }
    
    # Get the relevant attributes based on the essay type
    relevant_attributes = attribute_mapping.get(essay_type, [])
    
    # Display only the relevant attributes
    print(f"\nEssay Type: {essay_type}")
    for attr in relevant_attributes:
        print(f"{attr.capitalize().replace('_', ' ')}: {attributes[attr]}")
