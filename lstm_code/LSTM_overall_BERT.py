import pandas as pd
import numpy as np
from tensorflow.keras.layers import LSTM, Dense, Dropout
from transformers import TFBertModel, BertTokenizer
from keras.layers import LSTM, Dense, Dropout, Bidirectional, Input, Attention, GRU,  Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from keras.regularizers import l2
from keras.callbacks import EarlyStopping, TensorBoard, ReduceLROnPlateau
import tensorflow as tf
from sklearn.model_selection import KFold
from sklearn.metrics import cohen_kappa_score
import nltk
import datetime
from gensim.test.utils import datapath

nltk.download('stopwords')
nltk.download('punkt_tab')

import numpy as np
import nltk
import tensorflow as tf

data_path = '/mnt/c/Users/imana/Desktop/Masters/Foundations of Artificial Intelligence - AI701/AI_AES_project/dataset/asap++_data.csv'
X = pd.read_csv(data_path)
X = X.dropna()
print(X['essay'])

data = pd.DataFrame()
data['essay'] = X['essay']

y = np.round(X['domain1_score']) # max score 9=is 60
y = (y / 60) * 10 # we want scores to be from 0-10
data['overall_score'] = y

print(data)
print(y)

bert_model = TFBertModel.from_pretrained("bert-base-uncased")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
 
def get_model():
    # Define input for token IDs and attention masks
    input_ids = Input(shape=(None,), dtype=tf.int32, name="input_ids")
    attention_mask = Input(shape=(None,), dtype=tf.int32, name="attention_mask")

    # Define a Lambda layer to process the BERT model output
    def call_bert(inputs):
        input_ids, attention_mask = inputs
        outputs = bert_model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state  # Return the last hidden states

    # Specify the output shape explicitly
    bert_output = Lambda(
        lambda x: call_bert(x),
        output_shape=(None, 768),  # Output shape of BERT's hidden states
        name="bert_embedding"
    )([input_ids, attention_mask])

    # Shared LSTM layers after BERT
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, dropout=0.5, return_sequences=True))(bert_output)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, dropout=0.5))(x)
    x = Dropout(0.2)(x)

    # Output layer for predicting a single score using softmax
    # Target range is 0-10, we need 11 output classes
    output = Dense(11, activation='softmax', name='score')(x)

    # Define and compile the model
    model = Model(inputs=[input_ids, attention_mask], outputs=output)
    model.compile(
        loss='sparse_categorical_crossentropy',  # Loss for classification
        optimizer=Adam(learning_rate=0.0001),
        metrics=['accuracy']  # Use accuracy as the evaluation metric
    )

    model.summary()
    return model

cv = KFold(n_splits=5, shuffle=True)
results = []
y_pred_list = []

count = 1
for traincv, testcv in cv.split(data):
    print("\n--------Fold {}--------\n".format(count))
    X_test, X_train, y_test, y_train = data.iloc[testcv], data.iloc[traincv], y.iloc[testcv], y.iloc[traincv]

    train_essays = X_train['essay'].tolist()
    test_essays = X_test['essay'].tolist()

    # Step 1: Preprocess essays for BERT embeddings
    print("Preprocessing essays for BERT...")
    
    # Tokenize the training and testing essays
    train_tokens = tokenizer(
        train_essays,
        max_length=512,
        padding='max_length',
        truncation=True,
        return_tensors='tf'
    )
    test_tokens = tokenizer(
        test_essays,
        max_length=512,
        padding='max_length',
        truncation=True,
        return_tensors='tf'
    )

    train_input_ids = train_tokens['input_ids']
    train_attention_mask = train_tokens['attention_mask']
    test_input_ids = test_tokens['input_ids']
    test_attention_mask = test_tokens['attention_mask']

    # Step 2: Initialize the BiLSTM model
    lstm_model = get_model()

    # Tensorboard 
    log_dir = "workspace/AI_project/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

    # Early stopping callback
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    #L2 Regularization
    lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)

    # Train the model
    print("Training the model...")
    lstm_model.fit(
        [train_input_ids, train_attention_mask],
        y_train,
        batch_size=16,
        epochs=5,
        validation_data=([test_input_ids, test_attention_mask], y_test),
        callbacks=[early_stopping, tensorboard_callback, lr_scheduler]
    )

    # Predict on the test set
    print("Predicting on the test set...")
    y_pred = lstm_model.predict([test_input_ids, test_attention_mask])

    # Post-process predictions
    y_pred = np.round(y_pred).squeeze()  # Round to nearest integer
    print(f"Predicted values (rounded): {y_pred}")

    # Evaluate using Cohen's Kappa Score
    result = cohen_kappa_score(y_test.values, y_pred, weights='quadratic')
    print(f"Kappa Score: {result}")
    results.append(result)

    # Save the model for one of the folds
    if count == 5:
        lstm_model.save('./overall_score_lstm_bert.h5')

    count += 1

print("Average Kappa score after a 5-fold cross validation: ", np.round(np.array(results).mean(),decimals=4))