
# Automated Essay Scoring with Lightweight Pretrained Models and Attribute-Based Evaluation Using the ASAP++ Dataset

This repository provides an end-to-end framework for automated essay scoring (AES) using lightweight pre-trained models and attribute-based evaluation. It is built upon the ASAP++ dataset and supports various tasks such as argumentative attribute evaluation, score prediction, content-based assessment, and more.

---

## Repository Structure

### **Tree Structure**
```graphql
.
├── albert_ira/
│   ├── code_argumentative.ipynb       # Code for argumentative attribute modeling and evaluation
│   ├── code_dependent.ipynb           # Code for score-dependent attributes modeling and evaluation
│   ├── code_score.ipynb               # Code for overall score modeling and evaluation
│   ├── code_content.ipynb             # Code for content-specific attributes modeling and evaluation
│   ├── processed_essay_dataset.csv    # Preprocessed dataset from ASAP++ dataset
│   └── albert.py                      # Utility script with model architectures and helper functions
├── images/
│   ├── model_loss_plot.png            # Example: Loss plots from training
│   ├── performance_metrics_table.png  # Example: Metrics table for evaluation results
│   └── ...                            # Additional images generated during modeling and evaluation
├── lstm_code/
│   ├── LSTM_content_BERT.py           # LSTM implementation for content-based essay evaluation with BERT embeddings
│   ├── LSTM_content_training.py       # Training script for LSTM models (content evaluation)
│   ├── LSTM_overall_BERT.py           # LSTM model for overall score prediction using BERT embeddings
│   ├── multitask_script.py            # Multitask learning script for AES
│   └── overall_score_train.ipynb      # Notebook for overall score training and evaluation
├── result/
│   ├── essay_pipeline.py              # Essay Pipeline file
│   ├── albert_......pth               # Example: Trained model weights (PyTorch format)
│   ├── albert_......npy               # Example: Metadata file for model1
│   ├── lstm_........h5                # Example: Trained model weights (Keras/TensorFlow format)
│   └── ...                            # Additional trained models and metadata files
├── .gitignore                         # Git ignore file to exclude unnecessary files from version control
├── initial_exploration.ipynb          # Exploratory analysis and visualization of the dataset
└── RNN_classifier.ipynb               # RNN-based classifier for predicting essay types
```

## Detailed Folder Description

### **1. `albert_ira/`**
Contains implementations for various essay scoring tasks using ALBERT and other pretrained models.  

- **`code_argumentative.ipynb`**: 
  - Script for training and evaluating models for argumentative attributes such as organization, word choice, sentence fluency, and conventions.
  
- **`code_dependent.ipynb`**:
  - Script for training and evaluating models for score-dependent attributes like language, prompt adherence, and narrativity.

- **`code_score.ipynb`**:
  - Script for overall score modeling and evaluation.

- **`code_content.ipynb`**:
  - Script for content-specific attribute evaluation and modeling.

- **`processed_essay_dataset.csv`**:
  - The preprocessed dataset was derived from the ASAP++ dataset.

- **`albert.py`**:
  - Contains reusable utilities, such as model architectures, embedding processors, and training pipelines.

---

### **2. `images/`**
Stores visualizations and images generated during the modeling and evaluation processes.

---

### **3. `lstm_code/`**
Contains LSTM-based approaches for AES tasks.

- **`LSTM_content_BERT.py`**: LSTM implementation for content-based essay evaluation using BERT embeddings.
- **`LSTM_content_training.py`**: Script for training LSTM models for content evaluation.
- **`LSTM_overall_BERT.py`**: LSTM model for overall score prediction using BERT embeddings.
- **`multitask_script.py`**: Multitask learning script for simultaneous evaluation of multiple essay attributes.
- **`overall_score_train.ipynb`**: Notebook for training and evaluating overall score prediction models.

---

### **4. `result/`**
Folder containing trained model results:
- `.h5` and `.pth` files for storing model weights.
- `.npy` files for saving metadata such as embedding sizes.
- `essay_pipeline.py` is a script for running the complete essay pipeline, integrating with all trained model and its evaluation 
---

### **5. `.gitignore`**
Specifies files and folders to be excluded from version control, such as intermediate datasets, large models, and logs.

---

### **6. `initial_exploration.ipynb`**
Notebook for initial exploration and visualization of the ASAP++ dataset. Includes insights into data distributions, class imbalance, and feature analysis.

---

### **7. `RNN_classifier.ipynb`**
Notebook for building and training an RNN-based classifier for predicting essay types from the dataset.

---

## Usage

1. **Setup Environment**: Install dependencies and required libraries using `pip install -r requirements_albert.txt`.

2. **Run Preprocessing**:
   - Use the preprocessed dataset (`processed_essay_dataset.csv`) or preprocess the ASAP++ dataset manually.

3. **Train and Evaluate Models**:
   - Select the desired task and corresponding script (e.g., `code_argumentative.ipynb` for argumentative attribute modeling).
   - Configure hyperparameters and train the model.
   - Run the evaluation function to compute performance metrics of Quadratic Weighted Kappa (QWK) for the desired attribute.

---
## Team Members

- **Iman Andrea** [@AerdnaNami](https://github.com/AerdnaNami)
- **Amina Tariq** [@atzaffar](https://github.com/atzaffar)
- **Salsabila Zahirah** [@irasalsabila](https://github.com/irasalsabila)
---

## Acknowledgments

This project utilizes several datasets, pre-trained models, and research contributions. We acknowledge and cite the following works:

- **ASAP++ Dataset**:
  Sandeep Mathias and Pushpak Bhattacharyya. *ASAP++: Enriching the ASAP Automated Essay Grading Dataset with Essay Attribute Scores*. In Proceedings of the 11th International Conference on Language Resources and Evaluation, pages 1169-1173. Miyazaki, Japan. May 8-10, 2018. 
  

- **ALBERT**:
  Zhenzhong Lan, Mingda Chen, Sebastian Goodman, Kevin Gimpel, Piyush Sharma, and Radu Soricut. *ALBERT: A Lite BERT for Self-supervised Learning of Language Representations*. 2020. 
  

- **GloVe**:
  Jeffrey Pennington, Richard Socher, and Christopher Manning. *GloVe: Global Vectors for Word Representation*. Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 1532–1543, Doha, Qatar, October 2014.  
  
- **FastText**:
  Piotr Bojanowski, Edouard Grave, Armand Joulin, and Tomas Mikolov. *Enriching Word Vectors with Subword Information*. Transactions of the Association for Computational Linguistics, 2017.  

