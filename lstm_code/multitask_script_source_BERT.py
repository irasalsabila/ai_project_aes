import numpy as np
import pandas as pd 
from sklearn.model_selection import KFold
from sklearn.metrics import cohen_kappa_score, accuracy_score
from keras.callbacks import EarlyStopping, TensorBoard, ReduceLROnPlateau
from gensim.models import Word2Vec
import numpy as np
import datetime
import pandas as pd
import warnings
import nltk
from nltk.corpus import stopwords                   #Stopwords corpus

from gensim.models import Word2Vec                                   #For Word2Vec

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from keras.layers import LSTM, Dense, Dropout, Bidirectional, Input, Attention, GRU,  Lambda
from keras.models import Model
from keras.regularizers import l2
from tensorflow.keras.optimizers import Adam

import itertools
import re

from transformers import TFBertModel, BertTokenizer
import tensorflow as tf

snow = nltk.stem.SnowballStemmer('english')

warnings.filterwarnings("ignore")                    

train_data = 'dataset/asap++_data.csv'
df = pd.read_csv(train_data)

feature_1 = 'language'
feature_2 = 'prompt_adherence'
feature_3 = 'narrativity'

data = pd.DataFrame()
data['essay'] = df['essay']
data[feature_1] = df[feature_1]
data[feature_2] = df[feature_2]
data[feature_3] = df[feature_3]
data = data.dropna()

data_lang = []
data_prompt = []
data_nar = []

num = [1,2,3,4,5,6]
count = 0

def gen_num(num,length):
    num_list = [num]*length
    return num_list

def get_scores(feature):
    list_1 = []
    list_2 = []
    list_3 = []
    list_4 = []
    list_5 = []
    list_6 = []
    # getting essays that are the scores in num
    for i,j in zip(range(len(df)),df["essay"]):
        if df[feature][i] == 1: # essays scored 1
            list_1.append(j)
        if df[feature][i] == 2:
            list_2.append(j)
        if df[feature][i] == 3:
            list_3.append(j)
        if df[feature][i] == 4:
            list_4.append(j)
        if df[feature][i] == 5:
            list_5.append(j)
        if df[feature][i] == 6:
            list_6.append(j)

        data = list(itertools.chain(list_1,list_2,list_3,list_4,list_5,list_6))

    return list_1, list_2, list_3, list_4, list_5, list_6, data

list_1, list_2, list_3, list_4, list_5, list_6, data_lang = get_scores(feature = feature_1)
score_lang_1 = gen_num(1,len(list_1))
score_lang_2 = gen_num(2,len(list_2))
score_lang_3 = gen_num(3,len(list_3))
score_lang_4 = gen_num(4,len(list_4))
score_lang_5 = gen_num(5,len(list_5))
score_lang_6 = gen_num(6,len(list_6))
score_lang = list(itertools.chain(score_lang_1,score_lang_2,score_lang_3,score_lang_4,score_lang_5,score_lang_6))
print(len(score_lang), len(data_lang))

list_1, list_2, list_3, list_4, list_5, list_6, data_prompt = get_scores(feature=feature_2)
score_prompt_1 = gen_num(1,len(list_1))
score_prompt_2 = gen_num(2,len(list_2))
score_prompt_3 = gen_num(3,len(list_3))
score_prompt_4 = gen_num(4,len(list_4))
score_prompt_5 = gen_num(5,len(list_5))
score_prompt_6 = gen_num(6,len(list_6))
score_prompt = list(itertools.chain(score_prompt_1,score_prompt_2,score_prompt_3,score_prompt_4,score_prompt_5,score_prompt_6))
print(len(score_prompt), len(data_prompt))

list_1, list_2, list_3, list_4, list_5, list_6, data_nar = get_scores(feature=feature_3)
score_nar_1 = gen_num(1,len(list_1))
score_nar_2 = gen_num(2,len(list_2))
score_nar_3 = gen_num(3,len(list_3))
score_nar_4 = gen_num(4,len(list_4))
score_nar_5 = gen_num(5,len(list_5))
score_nar_6 = gen_num(6,len(list_6))
score_nar = list(itertools.chain(score_nar_1,score_nar_2,score_nar_3,score_nar_4,score_nar_5,score_nar_6))
print(len(score_nar), len(data_nar))

dictionary_lang = {'essay': data_lang, 'score_lang': score_lang}
dictionary_prompt = {'essay': data_prompt, 'score_prompt': score_prompt}
dictionary_nar = {'essay': data_nar, 'score_nar': score_nar}

df_lang = pd.DataFrame(dictionary_lang)
df_prompt = pd.DataFrame(dictionary_prompt)
df_nar = pd.DataFrame(dictionary_nar)

print(df_lang)
print(df_prompt)
print(df_nar)

stop_words = set(stopwords.words('english'))
len(stop_words) #finding stop words

# Load BERT model and tokenizer
bert_model = TFBertModel.from_pretrained("bert-base-uncased")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
 
def preprocess_for_bert(text_list, tokenizer, max_length=128):
    encoding = tokenizer(
        text_list,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors='tf'
    )
    return encoding['input_ids'], encoding['attention_mask']

def get_multi_task_taglm_model_arg():
    # Define input for token IDs and attention masks
    input_ids = Input(shape=(None,), dtype=tf.int32, name="input_ids")  # Sequence length
    attention_mask = Input(shape=(None,), dtype=tf.int32, name="attention_mask")

    # Define a Lambda layer to ensure compatibility with BERT model and specify output shape
    def call_bert(inputs):
        input_ids, attention_mask = inputs
        outputs = bert_model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state

    # Wrap BERT in a Lambda layer and specify the output shape
    bert_output = Lambda(lambda x: call_bert(x), output_shape=(None, 768))([input_ids, attention_mask])

    # Shared GRU layers after BERT
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, dropout=0.5, return_sequences=True))(bert_output)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, dropout=0.5))(x)
    x = Dropout(0.2)(x)

    # Output layer for each score type
    organization_score_output = Dense(1, activation='relu', name='org_score', kernel_regularizer=l2(0.01))(x)
    word_choice_score_output = Dense(1, activation='relu', name='word_score', kernel_regularizer=l2(0.01))(x)
    sentence_score_output = Dense(1, activation='relu', name='sentence_score', kernel_regularizer=l2(0.01))(x)
    conventions_score_output = Dense(1, activation='relu', name='conventions_score', kernel_regularizer=l2(0.01))(x)

    # Define the model with multiple outputs
    model = Model(inputs=[input_ids, attention_mask], outputs=[organization_score_output, word_choice_score_output, sentence_score_output, conventions_score_output])

    # Compile the model with a custom loss weight for each task if desired
    model.compile(
        loss={'org_score': 'mean_squared_error', 'word_score': 'mean_squared_error', 'sentence_score': 'mean_squared_error', 'conventions_score': 'mean_squared_error'},
        optimizer=Adam(learning_rate=0.0001),  # Lower learning rate for fine-tuning
        metrics={'org_score': 'mae', 'word_score': 'mae', 'sentence_score': 'mae', 'conventions_score': 'mae'}
    )

    model.summary()

    return model

def get_multi_task_taglm_model_source():
    # Define input for token IDs and attention masks
    input_ids = Input(shape=(None,), dtype=tf.int32, name="input_ids")  # Sequence length
    attention_mask = Input(shape=(None,), dtype=tf.int32, name="attention_mask")

    # Define a Lambda layer to ensure compatibility with BERT model and specify output shape
    def call_bert(inputs):
        input_ids, attention_mask = inputs
        outputs = bert_model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state

    # Wrap BERT in a Lambda layer and specify the output shape
    bert_output = Lambda(lambda x: call_bert(x), output_shape=(None, 768))([input_ids, attention_mask])

    # Shared LSTM layers after BERT
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, dropout=0.5, return_sequences=True))(bert_output)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, dropout=0.5))(x)
    x = Dropout(0.2)(x)

    # Output layers for each task
    lang_score_output = Dense(1, activation='relu', name='lang_score', kernel_regularizer=l2(0.01))(x)
    prompt_score_output = Dense(1, activation='relu', name='prompt_score', kernel_regularizer=l2(0.01))(x)
    nar_score_output = Dense(1, activation='relu', name='nar_score', kernel_regularizer=l2(0.01))(x)

    # Define the model with multiple outputs
    model = Model(inputs=[input_ids, attention_mask], outputs=[lang_score_output, prompt_score_output, nar_score_output])

    # Compile the model
    model.compile(
        loss={'lang_score': 'mean_squared_error', 'prompt_score': 'mean_squared_error', 'nar_score': 'mean_squared_error'},
        optimizer='adam',
        metrics={'lang_score': 'mae', 'prompt_score': 'mae', 'nar_score': 'mae'}
    )

    model.summary()

    return model


X = pd.concat([df_lang, df_prompt, df_nar], axis=1, join='inner')
X = X.loc[:,~X.columns.duplicated()].copy()

y_lang = df_lang['score_lang']
y_prompt = df_prompt['score_prompt']
y_nar = df_nar['score_nar']

y_combined = {
    'lang_score': y_lang,
    'prompt_score': y_prompt,
    'nar_score': y_nar
}

cv = KFold(n_splits=5, shuffle=True)
results = []
count = 1

for traincv, testcv in cv.split(X):
    print("\n--------Fold {}--------\n".format(count))

    # Split train and test sets for each task
    X_train, X_test = X.iloc[traincv], X.iloc[testcv]
    y_train = {key: y.iloc[traincv] for key, y in y_combined.items()}
    y_test = {key: y.iloc[testcv] for key, y in y_combined.items()}

    train_essays = X_train['essay'].tolist()
    test_essays = X_test['essay'].tolist()

    print("Preprocessing essays for BERT...")
    train_input_ids, train_attention_mask = preprocess_for_bert(train_essays, tokenizer, max_length=128)
    test_input_ids, test_attention_mask = preprocess_for_bert(test_essays, tokenizer, max_length=128)

    # Step 4: Initialize Multi-Task Model
    lstm_model = get_multi_task_taglm_model_arg()
    log_dir = "workspace/AI_project/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
    # Early stopping callback
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)

    # Step 5: Train the LSTM model with multi-task labels
    lstm_model.fit(
        [train_input_ids, train_attention_mask],
        y_train,
        validation_data=([test_input_ids, test_attention_mask], y_test),
        epochs=10,
        batch_size=16,
        callbacks=[early_stopping, tensorboard_callback, lr_scheduler]
    )


    # Save any one of the 5 models (for example, the last one)
    if count == 5:
        lstm_model.save('./models/lstm_sargumentative_multi_task_fold{}.h5'.format(count))

    # Step 6: Make Predictions on the Test Set
    test_input = tokenizer(
    test_essays,
    max_length=128,           # Define a maximum length for padding
    padding='max_length',      # Pad to the maximum length
    truncation=True,           # Truncate if any sequences are longer than max_length
    return_tensors='np'        # Return as NumPy arrays
    )

    test_input_ids = test_input['input_ids']        # Shape: (batch_size, sequence_length)
    test_attention_mask = test_input['attention_mask']

    # Create the attention mask based on the padding token (0)
    test_attention_mask = np.where(test_input_ids != 0, 1, 0)
    print("test_attention_mask:", test_attention_mask)  # Verify the output mask

    # Predict using both input_ids and attention_mask
    y_pred = lstm_model.predict([test_input_ids, test_attention_mask])


    # Step 7: Evaluate Kappa Scores for Each Output
    kappa_scores = {}
    for task in y_test.keys():
        y_pred_task = np.around(y_pred[list(y_test.keys()).index(task)]).flatten()  # Round predictions for each task
        kappa_scores[task] = cohen_kappa_score(y_test[task], y_pred_task, weights='quadratic')
        print(f"{task} Kappa Score: {kappa_scores[task]}")
    
    results.append(kappa_scores)
    count += 1
