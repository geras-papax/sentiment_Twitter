import numpy as np
import pandas as pd
import gensim
import re
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding
from keras.models import Sequential
from keras.layers import Dense,LSTM,Dropout

nltk.download('stopwords')
stop_words = stopwords.words('english')

def lstm(x_train, x_test, y_train,  y_test):
    #split the tweets
    texts = [text.split() for text in x_train]
    #building vocab and training word2vec model with the tweets
    w2v_model = gensim.models.word2vec.Word2Vec(vector_size=300, window=7, min_count=10, workers=8)
    w2v_model.build_vocab(texts)
    w2v_model.train(texts, total_examples=len(texts), epochs=10)
    #tokenize all the tweets
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(x_train)
    vocab_size=len(tokenizer.word_index)+1
    #uniform all the sequences in same length
    X_train = pad_sequences(tokenizer.texts_to_sequences(x_train), maxlen=300)
    X_test = pad_sequences(tokenizer.texts_to_sequences(x_test), maxlen=300)
    #creating embedding matrix
    embedding_matrix = np.zeros((vocab_size, 300))
    for word, i in tokenizer.word_index.items():
        if word in w2v_model.wv:
            embedding_matrix[i] = w2v_model.wv[word]
    embedding_layer = Embedding(vocab_size, 300, weights=[embedding_matrix], input_length=300, trainable=False)
    #building model 
    model = Sequential()
    model.add(embedding_layer)
    model.add(Dropout(0.5))
    model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(1, activation='relu'))
    print(model.summary())
    model.compile(loss='binary_crossentropy', optimizer="adam", metrics=['accuracy'])
    #train the model
    lstm_model = model.fit(X_train, y_train,batch_size=1024,epochs=15,validation_split=0.1,verbose=1)
    #show metrics
    acc = lstm_model.history['accuracy']
    val_acc = lstm_model.history['val_accuracy']
    loss = lstm_model.history['loss']
    val_loss = lstm_model.history['val_loss']
    epochs=range(len(acc))
    plt.plot(epochs,acc,label='Trainin_acc',color='blue')
    plt.plot(epochs,val_acc,label='Validation_acc',color='red')
    plt.title("Training and Validation Accuracy - Loss")
    plt.plot(epochs,loss,label='Training_loss',color='green')
    plt.plot(epochs,val_loss,label='Validation_loss',color='yellow')
    plt.legend()
    return tokenizer, model

def preprocess(tokenizer, text):
    review=re.sub('@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+',' ',text)
    review=review.lower()
    review=review.split()
    review=[word for word in review if not word in stop_words]
    print(review)
    review=pad_sequences(tokenizer.texts_to_sequences([review]), maxlen=300)
    return review

def prediction(model, tokenizer, review):
    review=preprocess(tokenizer, review)
    score=model.predict(review)
    score=score[0]
    if score<0.4:
        print("Negative")
    elif score>0.4 and score<0.6:
        print("Neutral")
    else:
        print("Positive")
    print(score)