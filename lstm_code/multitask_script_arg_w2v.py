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

feature_1 = 'word_choice'
feature_2 = 'organization'
feature_3 = 'sentence_fluency'
feature_4 = 'conventions'

data = pd.DataFrame()
data['essay'] = df['essay']
data[feature_1] = df[feature_1]
data[feature_2] = df[feature_2]
data[feature_3] = df[feature_3]
data[feature_4] = df[feature_4]
data = data.dropna()

data_org = []
data_word = []
data_sent = []
data_conv = []
data_lang = []
data_prompt = []
data_nar = []

num = [0,1,2,3,4,5,6]
count = 0

def gen_num(num,length):
    num_list = [num]*length
    return num_list

def get_scores(feature):
    list_0 = []
    list_1 = []
    list_2 = []
    list_3 = []
    list_4 = []
    list_5 = []
    list_6 = []
    # getting essays that are the scores in num
    for i,j in zip(range(len(df)),df["essay"]):
        if df[feature][i] == 0: # essays scored 1
            list_0.append(j)
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

        data = list(itertools.chain(list_0, list_1,list_2,list_3,list_4,list_5,list_6))

    return list_0, list_1, list_2, list_3, list_4, list_5, list_6, data

list_0, list_1, list_2, list_3, list_4, list_5, list_6, data_org = get_scores(feature = feature_1)
score_org_0 = gen_num(0,len(list_0))
score_org_1 = gen_num(1,len(list_1))
score_org_2 = gen_num(2,len(list_2))
score_org_3 = gen_num(3,len(list_3))
score_org_4 = gen_num(4,len(list_4))
score_org_5 = gen_num(5,len(list_5))
score_org_6 = gen_num(6,len(list_6))
score_org = list(itertools.chain(score_org_0,score_org_1,score_org_2,score_org_3,score_org_4,score_org_5,score_org_6))

list_0, list_1, list_2, list_3, list_4, list_5, list_6, data_word = get_scores(feature=feature_2)
score_word_0 = gen_num(0,len(list_0))
score_word_1 = gen_num(1,len(list_1))
score_word_2 = gen_num(2,len(list_2))
score_word_3 = gen_num(3,len(list_3))
score_word_4 = gen_num(4,len(list_4))
score_word_5 = gen_num(5,len(list_5))
score_word_6 = gen_num(6,len(list_6))
score_word = list(itertools.chain(score_word_0,score_word_1,score_word_2,score_word_3,score_word_4,score_word_5,score_word_6))

list_0, list_1, list_2, list_3, list_4, list_5, list_6, data_sent = get_scores(feature=feature_3)
score_sent_0 = gen_num(0,len(list_0))
score_sent_1 = gen_num(1,len(list_1))
score_sent_2 = gen_num(2,len(list_2))
score_sent_3 = gen_num(3,len(list_3))
score_sent_4 = gen_num(4,len(list_4))
score_sent_5 = gen_num(5,len(list_5))
score_sent_6 = gen_num(6,len(list_6))
score_sent = list(itertools.chain(score_sent_0,score_sent_1,score_sent_2,score_sent_3,score_sent_4,score_sent_5,score_sent_6))
print(len(score_sent), len(data_sent))

list_0, list_1, list_2, list_3, list_4, list_5, list_6, data_conv = get_scores(feature=feature_4)
score_conv_0 = gen_num(0,len(list_0))
score_conv_1 = gen_num(1,len(list_1))
score_conv_2 = gen_num(2,len(list_2))
score_conv_3 = gen_num(3,len(list_3))
score_conv_4 = gen_num(4,len(list_4))
score_conv_5 = gen_num(5,len(list_5))
score_conv_6 = gen_num(6,len(list_6))
score_conv = list(itertools.chain(score_conv_0,score_conv_1,score_conv_2,score_conv_3,score_conv_4,score_conv_5,score_conv_6))
print(len(score_conv), len(data_conv))

# dictionary of lists
dictionary_org = {'essay': data_org, 'score_org': score_org}
dictionary_word = {'essay': data_word, 'score_word': score_word}
dictionary_sent = {'essay': data_sent, 'score_sent': score_sent}
dictionary_conv = {'essay': data_conv, 'score_conv': score_conv}

df_org = pd.DataFrame(dictionary_org)
df_word = pd.DataFrame(dictionary_word)
df_sent = pd.DataFrame(dictionary_sent)
df_conv = pd.DataFrame(dictionary_conv)

print(df_org)
print(df_word)
print(df_sent)
print(df_conv)

nltk.download('stopwords')

stop_words = set(stopwords.words('english'))
len(stop_words) #finding stop words

corpus = []
for i in range(0, len(data)):
    review = re.sub('[^a-zA-Z]', ' ', data['essay'][i])
    review = review.lower()
    review = review.split()

    review = [snow.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)

voc_size=5000
onehot_repr=[one_hot(words,voc_size)for words in corpus]
type(onehot_repr)

sent_length=400
embedded_docs=pad_sequences(onehot_repr,padding='pre',maxlen=sent_length)


def essay_to_wordlist(essay_v, remove_stopwords):
    """Remove the tagged labels and word tokenize the sentence."""
    essay_v = re.sub("[^a-zA-Z]", " ", essay_v)
    words = essay_v.lower().split()
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]
    return (words)

def essay_to_sentences(essay_v, remove_stopwords):
    """Sentence tokenize the essay and call essay_to_wordlist() for word tokenization."""
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    raw_sentences = tokenizer.tokenize(essay_v.strip())
    sentences = []
    for raw_sentence in raw_sentences:
        if len(raw_sentence) > 0:
            sentences.append(essay_to_wordlist(raw_sentence, remove_stopwords))
    return sentences

def makeFeatureVec(words, model, num_features):
    """Make Feature Vector from the words list of an Essay."""
    featureVec = np.zeros((num_features,),dtype="float32")
    num_words = 0.
    try:
        index2word_set = set(model.wv.index_to_key)
    except(AttributeError):
        index2word_set = set(model.index_to_key)

    for word in words:
        if word in index2word_set:
            num_words += 1
            try:
                featureVec = np.add(featureVec,model.wv.get_vector(word))
            except(AttributeError):
                featureVec = np.add(featureVec,model.get_vector(word))

    featureVec = np.divide(featureVec,num_words)
    return featureVec

def getAvgFeatureVecs(essays, model, num_features):
    """Main function to generate the word vectors for word2vec model."""
    counter = 0
    essayFeatureVecs = np.zeros((len(essays),num_features),dtype="float32")
    for essay in essays:
        essayFeatureVecs[counter] = makeFeatureVec(essay, model, num_features)
        counter = counter + 1
    return essayFeatureVecs
   
def get_multitask_model():
    # Define input for token IDs and attention masks
    input_layer = Input(shape=(1, 300))  # Adjust input shape as necessary

    # Shared LSTM layers
    x = Bidirectional(LSTM(300, dropout=0.4, recurrent_dropout=0.4, return_sequences=True))(input_layer)
    x = Bidirectional(LSTM(64, recurrent_dropout=0.4))(x)
    x = Dropout(0.5)(x)

    # Output layer for each score type
    organization_score_output = Dense(1, activation='relu', name='org_score', kernel_regularizer=l2(0.01))(x)
    word_choice_score_output = Dense(1, activation='relu', name='word_score', kernel_regularizer=l2(0.01))(x)
    sentence_score_output = Dense(1, activation='relu', name='sentence_score', kernel_regularizer=l2(0.01))(x)
    conventions_score_output = Dense(1, activation='relu', name='conventions_score', kernel_regularizer=l2(0.01))(x)

    # Define the model with multiple outputs
    model = Model(inputs=input_layer, outputs=[organization_score_output, word_choice_score_output, sentence_score_output, conventions_score_output])

    # Compile the model with a custom loss weight for each task if desired
    model.compile(
        loss={'org_score': 'mean_squared_error', 'word_score': 'mean_squared_error', 'sentence_score': 'mean_squared_error', 'conventions_score': 'mean_squared_error'},
        optimizer=Adam(learning_rate=0.0001),  # Lower learning rate for fine-tuning
        metrics={'org_score': 'mae', 'word_score': 'mae', 'sentence_score': 'mae', 'conventions_score': 'mae'}
    )

    model.summary()

    return model


X = pd.concat([df_org, df_word, df_sent, df_conv], axis=1, join='inner')
X = X.loc[:,~X.columns.duplicated()].copy()

y_org = df_org['score_org']
y_word = df_word['score_word']
y_sent = df_sent['score_sent']
y_conv = df_conv['score_conv']

y_combined = {
    'org_score': y_org,
    'word_score': y_word,
    'sentence_score': y_sent,
    'conventions_score': y_conv
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

    # Step 1: Preprocess Essays for Word2Vec
    sentences = []
    for essay in train_essays:
        # Tokenize each essay into sentences
        sentences += essay_to_sentences(essay, remove_stopwords=True)

    # Check if sentences are populated
    if len(sentences) == 0:
        print("No sentences found in training data for this fold. Skipping fold.")
        continue  # Skip this fold if there are no sentences

    # Step 2: Initialize and Build Vocabulary for Word2Vec Model
    num_features = 300
    min_word_count = 1
    num_workers = 4
    context = 10
    downsampling = 1e-4

    print("Training Word2Vec Model...")
    w2v_model = Word2Vec(vector_size=num_features, min_count=min_word_count, workers=num_workers, window=context, sample=downsampling, sg=1)
    w2v_model.build_vocab(sentences)  # Explicitly build vocabulary
    w2v_model.train(sentences, total_examples=w2v_model.corpus_count, epochs=w2v_model.epochs)

    # Step 3: Generate Feature Vectors for Essays
    clean_train_essays = [essay_to_wordlist(essay, remove_stopwords=True) for essay in train_essays]
    clean_test_essays = [essay_to_wordlist(essay, remove_stopwords=True) for essay in test_essays]

    trainDataVecs = getAvgFeatureVecs(clean_train_essays, w2v_model, num_features)
    testDataVecs = getAvgFeatureVecs(clean_test_essays, w2v_model, num_features)

    # Reshape data for LSTM
    trainDataVecs = np.reshape(trainDataVecs, (trainDataVecs.shape[0], 1, trainDataVecs.shape[1]))
    testDataVecs = np.reshape(testDataVecs, (testDataVecs.shape[0], 1, testDataVecs.shape[1]))

    # Step 4: Initialize Multi-Task Model
    lstm_model = get_multitask_model()
    log_dir = "workspace/AI_project/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
    # Early stopping callback
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)

    # Step 5: Train the LSTM model with multi-task labels
    lstm_model.fit(
        trainDataVecs,
        y_train,
        validation_data=(testDataVecs, y_test),
        epochs=10,
        batch_size=16,
        callbacks=[early_stopping, tensorboard_callback, lr_scheduler]
    )


    # Save any one of the 5 models (for example, the last one)
    if count == 5:
        lstm_model.save('./models/lstm_argumentative_multi_task_fold{}.h5'.format(count))

    y_pred = lstm_model.predict(testDataVecs)
    y_pred = np.around(y_pred)

    # Step 7: Evaluate Kappa Scores for Each Output
    result = cohen_kappa_score(y_test.values, y_pred, weights='quadratic')
    acc = accuracy_score(y_test.values, y_pred)
    print("Accuracy Score: {}".format(acc))
    print("Kappa Score: {}".format(result))
    results.append(result)
    count += 1
