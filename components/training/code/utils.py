import os
import numpy as np

from keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

from sklearn.preprocessing import LabelEncoder

import matplotlib.pyplot as plt

def encodeLabels(y_train, y_test):
    """
    Encode the labels
    :param y_train: The training labels
    :param y_test: The testing labels
    :return: The encoded labels and the labels
    """

    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    y_test = le.transform(y_test)

    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    labels = le.classes_
    print(f"Labels: {labels} -- {le.transform(labels)}")

    return y_train, y_test, labels

def getEmbeddingMatrix(glove_file, num_tokens, tokenizer):
    """
    Get the embedding matrix
    :param glove_file: The glove file
    :param num_tokens: The number of the tokens
    :param tokenizer: The tokenizer
    :return: The embedding matrix
    """
    embedding_dim = 200
    hits = 0
    misses = 0

    embeddings_index = {}

    with open(glove_file, encoding="utf8") as f:
        for line in f:
            word, coefs = line.split(maxsplit=1)
            coefs = np.fromstring(coefs, "f", sep=" ")
            embeddings_index[word] = coefs
    print("Found %s word vectors." % len(embeddings_index))

    # Prepare embedding matrix
    embedding_matrix = np.zeros((num_tokens, embedding_dim))
    for word, i in tokenizer.word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # Words not found in embedding index will be all-zeros
            embedding_matrix[i] = embedding_vector
            hits += 1
        else:
            misses += 1
    print("Converted %d words (%d misses)" % (hits, misses))

    return embedding_matrix

def buildModel(input_lenght,vocabSize,embedding_matrix, classes, lr):
    """
    Build the model
    :param input_lenght: The length of the input
    :param vocabSize: The size of the vocabulary
    :param embedding_matrix: The embedding matrix
    :param classes: The amount of the classes
    :param lr: The learning rate
    :return: The model
    """

    adam = Adam(learning_rate=lr)

    model = Sequential()
    model.add(Embedding(vocabSize, 200, input_length=input_lenght, weights=[embedding_matrix], trainable=False))
    model.add(Bidirectional(LSTM(128, dropout=0.2,recurrent_dropout=0.2, return_sequences=True)))
    model.add(Bidirectional(LSTM(64, dropout=0.2,recurrent_dropout=0.2, return_sequences=True)))
    model.add(Bidirectional(LSTM(64, dropout=0.2,recurrent_dropout=0.2)))
    model.add(Dense(classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

    return model

def saveHistory(history, output_folder, data_name):
    """
    Save the history
    :param history: The history
    :param output_folder: The output folder
    :param data_name: The name of the dataset
    :return: None
    """
    # plot the loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.legend(['Train', 'Validation'], loc='upper right')
    plt.savefig(os.path.join(output_folder, data_name+'-loss.png'))
    plt.clf()

    # plot the accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epochs')
    plt.legend(['Train', 'Validation'], loc='upper right')
    plt.savefig(os.path.join(output_folder, data_name+'-accuracy.png'))
    plt.clf()