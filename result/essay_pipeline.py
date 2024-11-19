# %%
!pip install gensim tensorflow tf-keras





# %%
import os
import torch
import numpy as np
from tensorflow.keras.models import load_model
from transformers import TFBertModel, BertTokenizer
import tensorflow as tf
from transformers import AutoTokenizer, AutoModel
import torch.nn as nn
import torch.nn.functional as F
from gensim.models import KeyedVectors

# %%
# Load the pre-trained Word2Vec model
# Replace with your actual path to the Word2Vec model
w2v_model = KeyedVectors.load_word2vec_format("/home/amina.tariq/Desktop/nlp/word2vecmodel_overall_score.bin", binary=True)
print("Word2Vec model loaded successfully.")

def preprocess_and_embed(essay_text):
    # Tokenize essay text into words
    tokens = essay_text.lower().split()

    # Get embeddings for each word and average them to form a 300-dimensional vector
    embeddings = [w2v_model[word] for word in tokens if word in w2v_model]
    if embeddings:
        essay_embedding = np.mean(embeddings, axis=0)
    else:
        essay_embedding = np.zeros(300)  # Fallback if no embeddings are found

    # Reshape to (1, 1, 300) for the classifier input
    essay_embedding = essay_embedding.reshape(1, 1, 300)
    return essay_embedding

# Test the embedding function
sample_text = "This is an example essay text."
print("Sample embedding shape:", preprocess_and_embed(sample_text).shape)

# Load the BERT model and tokenizer
bert_model = TFBertModel.from_pretrained("bert-base-uncased")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Define the Lambda function used in model creation
def call_bert(inputs):
    # Unpack inputs and add error checks
    input_ids, attention_mask = inputs
    
    # Check if either input_ids or attention_mask is None
    if input_ids is None or attention_mask is None:
        raise ValueError("input_ids or attention_mask is None. Ensure both are defined and not None before passing to call_bert.")
    
    # Pass these tensors to BERT
    outputs = bert_model(input_ids=input_ids, attention_mask=attention_mask)
    return outputs.last_hidden_state





# %%
# Load the models
classifier = load_model('/home/amina.tariq/Desktop/nlp/rnn_classifier.h5')
lstm_overall_score = load_model('/home/amina.tariq/Desktop/nlp/lstm_overall_score.h5')
lstm_content_score = load_model('/home/amina.tariq/Desktop/nlp/lstm_content.h5')
lstm_argumentative_score = load_model('/home/amina.tariq/Desktop/nlp/lstm_argumentative_multi_task_fold5.h5')
lstm_source_dependent_score = load_model('/home/amina.tariq/Desktop/nlp/lstm_source_multi_task_fold5.h5')

print("All models loaded successfully.")

# %%
def pipeline(essay_text):
    # Step 1: Get classifier embeddings for essay type prediction
    embedding_300 = preprocess_and_embed(essay_text)
    essay_type_prediction = classifier.predict(embedding_300)
    essay_type = "Argumentative" if np.argmax(essay_type_prediction) == 1 else "Source-Dependent"
   
    # Step 2: Tokenize the essay text for BERT model input
    inputs = tokenizer(
        essay_text,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_tensors="tf"
    )
    input_ids = tf.cast(inputs['input_ids'], tf.int32)
    attention_mask = tf.cast(inputs['attention_mask'], tf.int32)

    # Debugging information
    print("Input IDs:", input_ids)
    print("Attention Mask:", attention_mask)
    print("Input IDs Shape:", input_ids.shape)
    print("Attention Mask Shape:", attention_mask.shape)

    # Step 3: Use the appropriate model based on essay type
    if essay_type == "Argumentative":
        multi_score = lstm_argumentative_score.predict({"input_ids": input_ids,"attention_mask": attention_mask})[0][0]

    else:
        multi_score = lstm_source_dependent_score.predict({"input_ids": input_ids,"attention_mask": attention_mask})[0][0]

    # Step 4: Get overall and content scores using the preprocessed 300-dimensional embeddings
    overall_score = lstm_overall_score.predict(embedding_300)[0][0]
    content_score = lstm_content_score.predict(embedding_300)[0][0]

    # Return results
    results = {
        "Essay Type": essay_type,
        "Overall Score": overall_score,
        "Content Score": content_score,
        # "Multi-Score": multi_score
    }
    return results


# %%
def load_glove_model(glove_file_path):
    embedding_dict = {}
    with open(glove_file_path, 'r', encoding="utf-8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            embedding_dict[word] = vector
    return embedding_dict

def load_fasttext_model(fasttext_file_path):
    model = KeyedVectors.load_word2vec_format(fasttext_file_path, binary=False)
    return {word: model[word] for word in model.index_to_key}

# %%
# Device Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %%
# Load ALBERT Tokenizer and Model
tokenizer = AutoTokenizer.from_pretrained("albert-base-v2")
albert_model = AutoModel.from_pretrained("albert-base-v2").to(device)

# %%
# MultiTask Model Definition
class MultiTaskModel(nn.Module):
    def __init__(self, input_shape):
        super(MultiTaskModel, self).__init__()
        self.fc1 = nn.Linear(input_shape, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(0.5)
        self.regression_head = nn.Linear(128, 1)
        self.task_uncertainty = nn.Parameter(torch.tensor([0.0, 0.0]), requires_grad=True)
        self.content_head = nn.Linear(128, 1)
        self.organization_head = nn.Linear(128, 1)
        self.word_choice_head = nn.Linear(128, 1)
        self.sentence_fluency_head = nn.Linear(128, 1)
        self.conventions_head = nn.Linear(128, 1)
        self.language_head = nn.Linear(128, 1) 
        self.prompt_adherence_head = nn.Linear(128, 1)  
        self.narrativity_head = nn.Linear(128, 1) 


    def forward(self, x):
        x = torch.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        x = torch.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        regression_output = self.regression_head(x)
        content_output = self.content_head(x)
        organization_output = self.organization_head(x)
        word_choice_output = self.word_choice_head(x)
        sentence_fluency_output = self.sentence_fluency_head(x)
        conventions_output = self.conventions_head(x)
        language_output = self.language_head(x) 
        prompt_adherence_output = self.prompt_adherence_head(x) 
        narrativity_output = self.narrativity_head(x)

        return (
            regression_output,
            content_output,
            organization_output,
            word_choice_output,
            sentence_fluency_output,
            conventions_output,
            language_output,
            prompt_adherence_output,
            narrativity_output,
        )
    
def preload_models(model_paths):
    """
    Preload all models for multiple tasks and embedding types into memory.

    Args:
        model_paths (dict): Nested dictionary with tasks as keys and sub-keys for embedding types.

    Returns:
        dict: Nested dictionary of loaded models, keyed by task and embedding type.
    """
    models = {}
    for task, emb_paths in model_paths.items():
        models[task] = {}
        for emb_type, path in emb_paths.items():
            # Load the state_dict
            state_dict = torch.load(path, map_location=torch.device('cpu'))

            # Dynamically determine input size from fc1.weight
            fc1_weight_shape = state_dict['fc1.weight'].shape
            input_shape = fc1_weight_shape[1]  # The second dimension gives the input size

            # Initialize the MultiTaskModel with the correct input shape
            model = MultiTaskModel(input_shape).to(device)

            # Get the current model's state_dict
            current_state_dict = model.state_dict()

            # Filter out unexpected keys from the state_dict
            filtered_state_dict = {k: v for k, v in state_dict.items() if k in current_state_dict}

            # Update the filtered state_dict with missing keys
            for key in current_state_dict.keys():
                if key not in filtered_state_dict:
                    print(f"Adding missing key: {key}")
                    filtered_state_dict[key] = current_state_dict[key]  # Use the default initialization

            # Load the filtered and updated state_dict into the model
            model.load_state_dict(filtered_state_dict)
            model.eval()

            # Store the model by embedding type
            models[task][emb_type] = model

    return models


# Generate ALBERT Embeddings
def get_albert_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=256).to(device)
    with torch.no_grad():
        outputs = albert_model(**inputs)
    return outputs.last_hidden_state[:, 0, :].cpu().numpy()

# Load Pretrained Word Embeddings (GloVe/FastText)
def load_word_embedding(text, embedding_dict):
    words = text.lower().split()
    vectors = [embedding_dict[word] for word in words if word in embedding_dict]
    if vectors:
        return np.mean(vectors, axis=0)
    return np.zeros(300)  # Default vector size for GloVe/FastText

# Create Combined Embedding (ALBERT + Optional Word Embeddings)
def create_combined_embedding(text, embedding_type=None, glove_model=None, fasttext_model=None):
    albert_emb = get_albert_embedding(text).flatten()

    if embedding_type == "glove":
        additional_emb = load_word_embedding(text, glove_model)
    elif embedding_type == "fasttext":
        additional_emb = load_word_embedding(text, fasttext_model)
    else:
        additional_emb = np.array([])

    albert_emb_tensor = torch.tensor(albert_emb, dtype=torch.float32).to(device)

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

    return combined_emb.cpu().numpy()

# %%
# Unified Pipeline for Multiple Tasks and Embedding Types
def unified_pipeline(essay_text, all_models, glove_model=None, fasttext_model=None):
    """
    Process essay text using preloaded models for multiple tasks and embeddings.

    Args:
        essay_text (str): Input essay text.
        all_models (dict): Nested dictionary of preloaded models keyed by task and embedding type.
        glove_model (dict, optional): Preloaded GloVe embeddings.
        fasttext_model (dict, optional): Preloaded FastText embeddings.

    Returns:
        dict: Combined results for all tasks and embedding types.
    """
    results = {}

    for task, models in all_models.items():
        task_results = {}

        for emb_type, model in models.items():
            if emb_type == "albert":
                embedding = create_combined_embedding(essay_text, embedding_type=None)
            elif emb_type == "glove":
                embedding = create_combined_embedding(essay_text, embedding_type="glove", glove_model=glove_model)
            elif emb_type == "fasttext":
                embedding = create_combined_embedding(essay_text, embedding_type="fasttext", fasttext_model=fasttext_model)
            else:
                continue  # Skip unknown types

            embedding_tensor = torch.tensor(embedding, dtype=torch.float32).to(device).unsqueeze(0)

            # Get Predictions
            with torch.no_grad():
                outputs = model(embedding_tensor)

            # The first output (regression_output) is used for scoring
            regression_output = outputs[0]  # Extract the first element from the tuple
            task_results[f"{emb_type.capitalize()}"] = regression_output.cpu().item()

        # Store results for the current task
        results[task.replace("_", " ").capitalize()] = task_results


    return results


# Example Usage
if __name__ == "__main__":
    # Define paths for each task and embedding type
    model_paths = {
        "overall_scoring": {
            "albert": "/home/amina.tariq/Desktop/nlp/albert3_model_albert.pth",
            "glove": "/home/amina.tariq/Desktop/nlp/albert3_model_glove.pth",
            "fasttext": "/home/amina.tariq/Desktop/nlp/albert3_model_fasttext.pth",
        },
        "content_scoring": {
            "albert": "/home/amina.tariq/Desktop/nlp/albert4_model_albert.pth",
            "glove": "/home/amina.tariq/Desktop/nlp/albert4_model_glove.pth",
            "fasttext": "/home/amina.tariq/Desktop/nlp/albert4_model_fasttext.pth",
        },
        "argumentative_scoring": {
            "albert": "/home/amina.tariq/Desktop/nlp/albert6_model_albert.pth",
            "glove": "/home/amina.tariq/Desktop/nlp/albert6_model_glove.pth",
            "fasttext": "/home/amina.tariq/Desktop/nlp/albert7_model_fasttext.pth",
        },
        "source_dependency_scoring": {
            "albert": "/home/amina.tariq/Desktop/nlp/albert7_model_albert.pth",
            "glove": "/home/amina.tariq/Desktop/nlp/albert7_model_glove.pth",
            "fasttext": "/home/amina.tariq/Desktop/nlp/albert7_model_fasttext.pth",
        },
    }

    
    glove_model = load_glove_model("/home/amina.tariq/Desktop/nlp/glove.6B.300d.txt")
    fasttext_model = load_fasttext_model("/home/amina.tariq/Desktop/nlp/wiki.en.vec")

    # Preload Models
    all_models = preload_models(model_paths)
    # Debug the loaded models
    for task, emb_models in all_models.items():
        print(f"\nTask: {task}")
        for emb_type, model in emb_models.items():
            print(f"  Embedding Type: {emb_type}, Model Type: {type(model)}")

# %%
# Unified Pipeline for ALBERT and LSTM Models
def combined_pipeline(essay_text, all_models, glove_model=None, fasttext_model=None):
    """
    Process essay text using ALBERT and LSTM models.

    Args:
        essay_text (str): Input essay text.
        all_models (dict): Nested dictionary of preloaded ALBERT models keyed by task and embedding type.
        glove_model (dict, optional): Preloaded GloVe embeddings.
        fasttext_model (dict, optional): Preloaded FastText embeddings.

    Returns:
        dict: Combined results from ALBERT models and LSTM models.
    """
    # Get results from the ALBERT pipeline
    albert_results = unified_pipeline(essay_text, all_models, glove_model=glove_model, fasttext_model=fasttext_model)

    # Get results from the LSTM pipeline
    lstm_results = pipeline(essay_text)

    # Combine the results
    combined_results = {
        "ALBERT Models": albert_results,
        "LSTM Models": lstm_results,
    }

    return combined_results


# Example Usage
if __name__ == "__main__":
    # Test the combined pipeline
    test_essay = "This is an example essay for testing purposes."
    combined_results = combined_pipeline(test_essay, all_models, glove_model=glove_model, fasttext_model=fasttext_model)

    # Print combined results
    print("\nCombined Pipeline Results:")
    for section, section_results in combined_results.items():
        print(f"\n{section}:")
        if isinstance(section_results, dict):  # Ensure section_results is a dictionary
            for task, task_results in section_results.items():
                if isinstance(task_results, dict):  # Ensure task_results is a dictionary
                    print(f"\n{task}:")
                    for emb_type, score in task_results.items():
                        print(f"  {emb_type}: {score:.2f}")
                else:
                    print(f"  {task}: {task_results}")
        else:
            print(f"  {section}: {section_results}")


# %%
# Test the pipeline
test_essay = "This is an example essay for testing purposes."
results = unified_pipeline(test_essay, all_models, glove_model=glove_model, fasttext_model=fasttext_model)

# Debugging: Print structure of results
print("\nPipeline Results:")
if isinstance(results, dict):  # Ensure results is a dictionary
    for task, task_results in results.items():
        if isinstance(task_results, dict):  # Ensure task_results is a dictionary
            print(f"\n{task}:")
            for emb_type, score in task_results.items():
                print(f"  {emb_type}: {score:.2f}")
        else:
            print(f"Unexpected structure for task '{task}':", task_results)
else:
    print("Unexpected structure for results:", results)



