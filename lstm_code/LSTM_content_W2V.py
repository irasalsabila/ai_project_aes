import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import pandas as pd

import warnings
warnings.filterwarnings("ignore")                     #Ignoring unnecessory warnings

import numpy as np                                  #for large and multi-dimensional arrays
import pandas as pd                                 #for data manipulation and analysis
import nltk                                         #Natural language processing tool-kit

from nltk.corpus import stopwords                   #Stopwords corpus
from nltk.stem import PorterStemmer                 # Stemmer
import re

from sklearn.feature_extraction.text import TfidfVectorizer          #For TF-IDF
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import cohen_kappa_score
from gensim.models import Word2Vec                                   #For Word2Vec

from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense

from keras.layers import LSTM, Dense, Dropout
from keras.models import Sequential
import keras.backend as K
import itertools

nltk.download('stopwords')

stop_words = set(stopwords.words('english'))
len(stop_words) #finding stop words

snow = nltk.stem.SnowballStemmer('english')

train_data = '/home/iman.alsikaiti/workspace/AI_project/dataset/asap++_data.csv'
df = pd.read_csv(train_data)
feature = "content"

data = pd.DataFrame()
data['essay'] = df['essay']
data[feature] = df[feature]
data = data.dropna()

list_0 = []
list_1 = []
list_2 = []
list_3 = []
list_4 = []
list_5 = []
list_6 = []

data = []
num = [0,1,2,3,4,5,6]
count = 0

# getting essays that are the scores in num
for i,j in zip(range(len(df)),df["essay"]):
    if df[feature][i] == 0: # essays scored 0
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


data = list(itertools.chain(list_0,list_1,list_2,list_3,list_4,list_5,list_6))
print(data)

def gen_num(num,length):
    num_list = [num]*length
    return num_list

score_0 = gen_num(0,len(list_0))
score_1 = gen_num(1,len(list_1))
score_2 = gen_num(2,len(list_2))
score_3 = gen_num(3,len(list_3))
score_4 = gen_num(4,len(list_4))
score_5 = gen_num(5,len(list_5))
score_6 = gen_num(6,len(list_6))

score = list(itertools.chain(score_0,score_1,score_2,score_3,score_4,score_5,score_6))

dictnary = {'essay': data, 'score': score}
df = pd.DataFrame(dictnary)
print(df)

corpus = []
for i in range(0, len(df)):
    review = re.sub('[^a-zA-Z]', ' ', df['essay'][i])
    review = review.lower()
    review = review.split()

    review = [snow.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)

voc_size=5000
onehot_repr=[one_hot(words,voc_size)for words in corpus]

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
    
def get_model():
    """Define the model."""
    model = Sequential()
    model.add(LSTM(300, dropout=0.4, recurrent_dropout=0.4, input_shape=[1, 300], return_sequences=True))
    model.add(LSTM(64, recurrent_dropout=0.4))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='relu'))

    model.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=['accuracy','mae'])
    model.summary()

    return model

X=df
y = X['score']

cv = KFold(n_splits = 5, shuffle = True)
results = []
y_pred_list = []

count = 1
for traincv, testcv in cv.split(X):
    print("\n--------Fold {}--------\n".format(count))
    X_test, X_train, y_test, y_train = X.iloc[testcv], X.iloc[traincv], y.iloc[testcv], y.iloc[traincv]

    train_essays = X_train['essay']
    test_essays = X_test['essay']

    sentences = []

    for essay in train_essays:
            # Obtaining all sentences from the training essays.
            sentences += essay_to_sentences(essay, remove_stopwords = True)

    # Initializing variables for word2vec model.
    num_features = 300
    min_word_count = 40
    num_workers = 4
    context = 10
    downsampling = 1e-3

    print("Training Word2Vec Model...")
    model = Word2Vec(sentences, workers=num_workers, vector_size=num_features, min_count = min_word_count, window = context, sample = downsampling)

    model.init_sims(replace=True)
    model.wv.save_word2vec_format('workspace/AI_project/model/word2vecmodel_content.bin', binary=True)

    clean_train_essays = []

    # Generate training and testing data word vectors.
    for essay_v in train_essays:
        clean_train_essays.append(essay_to_wordlist(essay_v, remove_stopwords=True))
    trainDataVecs = getAvgFeatureVecs(clean_train_essays, model, num_features)

    clean_test_essays = []
    for essay_v in test_essays:
        clean_test_essays.append(essay_to_wordlist( essay_v, remove_stopwords=True ))
    testDataVecs = getAvgFeatureVecs( clean_test_essays, model, num_features )

    trainDataVecs = np.array(trainDataVecs)
    testDataVecs = np.array(testDataVecs)
    # Reshaping train and test vectors to 3 dimensions. (1 represnts one timestep)
    trainDataVecs = np.reshape(trainDataVecs, (trainDataVecs.shape[0], 1, trainDataVecs.shape[1]))
    testDataVecs = np.reshape(testDataVecs, (testDataVecs.shape[0], 1, testDataVecs.shape[1]))

    lstm_model = get_model()
    lstm_model.fit(trainDataVecs, y_train, batch_size=64, epochs=50)
    y_pred = lstm_model.predict(testDataVecs)

    # Save any one of the 5 models.
    if count == 5:
         lstm_model.save('workspace/AI_project/model/lstm_content.h5')

    # Round y_pred to the nearest integer.
    y_pred = np.around(y_pred)

    # Evaluate the model on the evaluation metric. "Quadratic mean averaged Kappa"
    result = cohen_kappa_score(y_test.values,y_pred,weights='quadratic')
    from sklearn.metrics import accuracy_score
    acc = accuracy_score(y_test.values,y_pred)
    print("acc Score: {}".format(acc))
    print("Kappa Score: {}".format(result))
    results.append(result)

    count += 1


print("Average Kappa score after a 5-fold cross validation: ",np.around(np.array(results).mean(),decimals=4))
