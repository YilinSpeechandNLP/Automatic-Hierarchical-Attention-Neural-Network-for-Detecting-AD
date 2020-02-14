import numpy as np
import re
import csv
from bs4 import BeautifulSoup
import os
import tables
from sklearn.preprocessing import StandardScaler
from nltk.corpus import stopwords
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.layers.normalization import BatchNormalization
from keras.utils.np_utils import to_categorical
from keras.layers import Dense, Input,Lambda, Add, average,concatenate
from keras.layers import Conv1D, MaxPooling1D, Embedding, Dropout, GRU, Bidirectional, TimeDistributed, LSTM
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras.engine.topology import Layer
from keras import initializers
from keras import regularizers
from nltk import tokenize
import warnings
import pdb
import copy
from keras import backend as K
from sklearn.metrics import precision_recall_fscore_support
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
K.set_learning_phase(1)
train_path='text/dementiabank.hdf5'


MAX_SENT_LENGTH = 30
MAX_SENTS = 30
MAX_NB_WORDS = 1700
LSTM_DIM = 100
EMBEDDING_DIM=100
DENSE_DIM=50
Att_DIM=30
batch_size=20
fold_CV=10
Drop_rate=0.3
epochs = 20
class_num=2
print('parameters:\n********************************')
print('EMBEDDING_DIM:',str(EMBEDDING_DIM))
print('MAX_SENT_LENGTH:',str(MAX_SENT_LENGTH))
print('MAX_SENTS:',str(MAX_SENTS))
print('MAX_NB_WORDS:',str(MAX_NB_WORDS))
print('LSTM_DIM:',str(LSTM_DIM))
print('DENSE_DIM:',str(DENSE_DIM))
print('batch_size:',str(batch_size))
print('Drop_rate:',str(Drop_rate))
print('Att_DIM:',str(Att_DIM))
print('epochs:',str(epochs))
print('class_num',str(class_num))
print('********************************')
def check_units(y_true, y_pred):
    if y_pred.shape[1] != 1:
      y_pred = y_pred[:,1:2]
      y_true = y_true[:,1:2]
    return y_true, y_pred

def pr(y_true, y_pred):
    y_true, y_pred = check_units(y_true, y_pred)
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def rc(y_true, y_pred):
    y_true, y_pred = check_units(y_true, y_pred)
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    y_true, y_pred = check_units(y_true, y_pred)
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


def clean_str(string):
    """
    Tokenization/string cleaning for dataset
    Every dataset is lower cased except
    """
    string = re.sub(r"\\", "", string)
    string = re.sub(r"\'", "", string)
    string = re.sub(r"\"", "", string)
    return string.strip().lower()

def data_split(i,x,y,names):

    def set_change(x_f,y_f,n_f,x_b,y_b,n_b):
        while n_f[-1][:5] == n_b[0][:5]:
            x_f = np.concatenate((x_f, x_b[0][np.newaxis,:]),axis=0)
            y_f = np.concatenate((y_f, y_b[0][np.newaxis,:]),axis=0)
            n_f = np.concatenate((n_f,[n_b[0]]),axis=0)
            x_b = x_b[1:]
            y_b = y_b[1:]
            n_b = n_b[1:]
        return x_f,y_f,n_f,x_b,y_b,n_b

    x_val=x[i*n_segment:(i+1)*n_segment]
    y_val=y[i*n_segment:(i+1)*n_segment]
    n_val=names[i*n_segment:(i+1)*n_segment]
    if i==fold_CV-2:
        x_test=x[(i + 1) %fold_CV *n_segment :]
        y_test=y[(i + 1) %fold_CV *n_segment:]
        n_test = names[(i + 1) %fold_CV *n_segment:]
        x_train=x[:i*n_segment]
        y_train=y[:i*n_segment]
        n_train=names[:i*n_segment]
    elif i==fold_CV-1:
        x_test=x[:n_segment]
        y_test=y[:n_segment]
        n_test=names[:n_segment]
        x_train=x[(i + 2) %fold_CV * n_segment:i*n_segment]
        y_train=y[(i + 2) % fold_CV * n_segment:i * n_segment]
        n_train=names[(i + 2) % fold_CV * n_segment:i * n_segment]
    else:
        x_test=x[(i+1)*n_segment:(i+2)*n_segment]
        y_test=y[(i+1)*n_segment:(i+2)*n_segment]
        n_test=names[(i+1)*n_segment:(i+2)*n_segment]
        x_train=np.concatenate((x[(i+2)*n_segment:],x[:i*n_segment]),axis=0)
        y_train = np.concatenate((y[(i+2)*n_segment:],y[:i*n_segment]),axis=0)
        n_train =  np.concatenate((names[(i+2)*n_segment:],names[:i*n_segment]),axis=0)

    x_val, y_val, n_val,x_test,y_test,n_test=set_change(x_val,y_val,n_val,x_test,y_test,n_test)
    x_test, y_test, n_test, x_train, y_train, n_train = set_change(x_test, y_test, n_test, x_train,y_train, n_train)
    x_train,y_train, n_train, x_val, y_val, n_val = set_change(x_train,y_train, n_train, x_val, y_val, n_val)

    return x_val,y_val,n_val,x_train,y_train,n_train,x_test,y_test,n_test

def data_solved(fileh):
    reviews=[]
    labels = []
    texts = []
    names=[]

    for idx in range(fileh.root.content.shape[0]):
        text = BeautifulSoup(fileh.root.content[idx].decode('utf-8'),features="html.parser")
        text = clean_str(text.get_text())
        texts.append(text)
        sentences = tokenize.sent_tokenize(text)
        reviews.append(sentences)
        labels.append(fileh.root.label[idx])
        names.append(fileh.root.name[idx])

    return texts,labels,reviews,names

def set_change(x_f,y_f,n_f,x_b,y_b,n_b):
    if y_f!=[] and y_b!=[]:
        while n_f[-1][:5] == n_b[0][:5]:
            x_f = np.concatenate((x_f, x_b[0][np.newaxis,:]),axis=0)
            y_f = np.concatenate((y_f, y_b[0][np.newaxis,:]),axis=0)
            n_f = np.concatenate((n_f,[n_b[0]]),axis=0)
            x_b = x_b[1:]
            y_b = y_b[1:]
            n_b = n_b[1:]
    return x_f,y_f,n_f,x_b,y_b,n_b

def data_solved_2(texts,labels,reviews):
    data = np.zeros((len(texts), MAX_SENTS, MAX_SENT_LENGTH), dtype='int32')

    for i, sentences in enumerate(reviews):
        for j, sent in enumerate(sentences):
            if j < MAX_SENTS:
                wordTokens = text_to_word_sequence(sent)
                #wordTokens = [word for word in wordTokens if word not in stop_words]
                k = 0
                for _, word in enumerate(wordTokens):
                    if k < MAX_SENT_LENGTH and tokenizer.word_index[word] < MAX_NB_WORDS:
                        data[i, j, k] = tokenizer.word_index[word]
                        k = k + 1

    labels = to_categorical(np.asarray(labels))
    print(('Shape of data tensor:', data.shape))
    print(('Shape of label tensor:', labels.shape))

    return data,labels



class AttLayer(Layer):
    def __init__(self, attention_dim):
        self.init = initializers.get('normal')
        self.supports_masking = True
        self.attention_dim = attention_dim
        super(AttLayer, self).__init__()

    def build(self, input_shape):
        assert len(input_shape) == 3
        self.W = K.variable(self.init((input_shape[-1], self.attention_dim)))
        self.b = K.variable(self.init((self.attention_dim, )))
        self.u = K.variable(self.init((self.attention_dim, 1)))
        self.trainable_weights = [self.W, self.b, self.u]
        super(AttLayer, self).build(input_shape)

    def compute_mask(self, inputs, mask=None):
        return mask

    def call(self, x, mask=None):
        uit = K.tanh(K.bias_add(K.dot(x, self.W), self.b))
        ait = K.dot(uit, self.u)
        ait = K.squeeze(ait, -1)

        ait = K.exp(ait)

        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            ait *= K.cast(mask, K.floatx())
        ait /= K.cast(K.sum(ait, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        ait = K.expand_dims(ait)
        weighted_input = x * ait
        output = K.sum(weighted_input, axis=1)

        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])

def H_Model():
    ############word-level#############
    sentence_input = Input(shape=(MAX_SENT_LENGTH,), dtype='float32')
    x = embedding_layer(sentence_input)
    x = Dropout(Drop_rate)(x)
    #x = Dropout(Drop_rate)(x)
    x = Bidirectional(LSTM(LSTM_DIM, return_sequences=True),merge_mode='sum')(x)
    x = Dropout(Drop_rate)(x)
    x = Dense(DENSE_DIM, activation='relu')(x)
    #x = Dropout(Drop_rate)(x)
    output = AttLayer(Att_DIM)(x)
    sentEncoder = Model(sentence_input, output)
    print(sentEncoder.summary())
    ############sentence-level: Bi-rnn############
    review_input = Input(shape=(MAX_SENTS, MAX_SENT_LENGTH), dtype='float32')
    x = TimeDistributed(sentEncoder)(review_input)
    x = Bidirectional(GRU(LSTM_DIM, return_sequences=True),merge_mode='sum')(x)
    x = Dropout(Drop_rate)(x)
    x = AttLayer(Att_DIM)(x)
    #x = Dropout(Drop_rate)(x)
    preds = Dense(class_num, activation='sigmoid', kernel_regularizer=regularizers.l2(0.0001))(x)
    #x = Dropout(Drop_rate)(x)
    model = Model(review_input, preds)
    print(model.summary())
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',#''rmsprop',
                  metrics=['accuracy'])
    return model

fileh=tables.open_file(train_path,mode='r')
tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
#stop_words = set(stopwords.words('english'))
texts,labels,reviews,names=data_solved(fileh)
print((" Data shape:", fileh.root.content.shape))
tokenizer.fit_on_texts(texts)
word_index = tokenizer.word_index
print(('Total %s unique tokens.' % len(word_index)))

GLOVE_DIR = "glove"
embeddings_index = {}
f = open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt'))
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()
x, y=data_solved_2(texts,labels,reviews)
n_segment=int(len(labels)/fold_CV)
accuracy=[]
for i in range(fold_CV):
    np.random.seed(i)
    embedding_matrix = np.random.random((len(word_index) + 1, EMBEDDING_DIM))
    for word, j in list(word_index.items()):
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[j] = embedding_vector

    embedding_layer = Embedding(len(word_index) + 1,
                                EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                input_length=MAX_SENT_LENGTH,
                                trainable=True,
                                mask_zero=True)

    print(("fold:"+str(i)+"- model fitting - Hierachical attention network"))

    x_val, y_val, n_val, x_train, y_train, n_train, x_test, y_test, n_test = data_split(i, x, y, names)
    
    path="model_2/folder-"+str(i)
    print(path)
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)

    # save the best model for test part
    filepath = path+"/best_weights.h5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True,
                                 mode='max',save_weights_only=True)
    model=H_Model()
    model.fit(x_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  validation_data=(x_val, y_val),
                  callbacks=[checkpoint],
                  shuffle=True)
    model.load_weights(filepath)
    loss, acc = model.evaluate(x_test, y_test, batch_size=batch_size)
    print("test loss:", loss, " test acc:", acc)
    accuracy.append(acc)
accuracy=np.asarray(accuracy)
acc_mean = np.mean(accuracy, axis=0)
print("average accuracy:", acc_mean)
