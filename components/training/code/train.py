import os
import argparse
import pandas as pd
import numpy as np

from sklearn.metrics import classification_report, confusion_matrix

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

from utils import *

PATIENCE = 5
VALSPLIT = 0.2
BATCHSIZE = 256
def main():
    global PATIENCE, VALSPLIT, BATCHSIZE
    parser = argparse.ArgumentParser(description='Preprocess the dataset')
    parser.add_argument('--training_data', type=str, help='Path to the training dataset folder')
    parser.add_argument('--testing_data', type=str, help='Path to the training dataset folder')
    parser.add_argument('--data_name', type=str, help='Name of the dataset')
    parser.add_argument('--epochs', type=int, help='Amount of the epochs for training')
    parser.add_argument('--batchsize', type=int, help='The batch size for training', default=BATCHSIZE)
    parser.add_argument('--valsplit', type=int, help='The validation split for training must be between 0 and 100', default=VALSPLIT)
    parser.add_argument('--patience', type=int, help='The patience for training for lowering Learning rate', default=PATIENCE)
    parser.add_argument('--output_folder', type=str, help='Output folder for the model')
    parser.add_argument('--glove', type=str, help='Name of the glove file')
    parser.add_argument('--data_folder', type=str, help='Path to the data folder')
    args = parser.parse_args()

    training_data = args.training_data
    testing_data = args.testing_data
    data_name = args.data_name
    epochs = args.epochs
    output_folder = args.output_folder
    data_folder = args.data_folder
    glove_file = args.glove
    VALSPLIT = args.valsplit / 100
    BATCHSIZE = args.batchsize
    PATIENCE = args.patience

    # read the csv files
    training_df = pd.read_csv(os.path.join(training_data, data_name+'-training.csv'))
    testing_df = pd.read_csv(os.path.join(testing_data, data_name+'-testing.csv'))
    print(f"Training data: {training_df.shape}")
    print(f"Testing data: {testing_df.shape}")

    # get the training and testing data
    X_train = training_df['content'].astype(str)
    y_train = training_df['sentiment']

    X_test = testing_df['content'].astype(str)
    y_test = testing_df['sentiment']

    print(f"Training data: {X_train.shape} -- {y_train.shape}")
    print(f"Testing data: {X_test.shape} -- {y_test.shape}")

    # encode the labels
    y_train, y_test, labels = encodeLabels(y_train, y_test)

    # tokenize the data
    tokenizer = Tokenizer(oov_token='UNK')
    full_text = pd.concat([X_train, X_test])
    tokenizer.fit_on_texts(full_text)

    # get the vocabulary size
    vocab_size = len(tokenizer.word_index) + 1
    print(f"Vocabulary size: {vocab_size}")

    # get the max length of the sequence
    max_length = max([len(s.split()) for s in full_text])

    # get the sequences
    sequences_train = tokenizer.texts_to_sequences(X_train)
    sequences_test = tokenizer.texts_to_sequences(X_test)

    # pad the sequences
    X_train = pad_sequences(sequences_train, maxlen=max_length, padding='post')
    X_test = pad_sequences(sequences_test, maxlen=max_length, padding='post')

    # get the embedding matrix
    path_to_glove_file = os.path.join(data_folder, glove_file)
    embedding_matrix = getEmbeddingMatrix(path_to_glove_file, vocab_size, tokenizer)

    # get the model
    model = buildModel(X_train.shape[1], vocab_size, embedding_matrix, len(labels), 0.0001)

    model_path = os.path.join(output_folder, data_name)
    os.makedirs(model_path, exist_ok=True)
    model_name = os.path.join(model_path, data_name)
    # Creating callbacks for early stopping, reducing the learning rate and saving the best model
    modelCheckpoint = ModelCheckpoint(filepath=model_name+'.h5', 
                                      monitor='val_loss', 
                                      verbose=1, 
                                      save_best_only=True)
    
    earlyStopping = EarlyStopping(monitor='val_loss', 
                                  patience=PATIENCE, 
                                  verbose=1, 
                                  restore_best_weights=True)
    
    reduce_lr = ReduceLROnPlateau(factor=0.5, 
                                  patience=PATIENCE, 
                                  verbose=1)
    
    # train the model
    print("Training the model")
    history = model.fit(X_train, y_train, 
                        epochs=epochs, 
                        validation_split=VALSPLIT, 
                        callbacks=[modelCheckpoint, earlyStopping, reduce_lr],
                        batch_size=BATCHSIZE,
                        verbose=1)
    
    # save the history
    print("Saving the history")
    saveHistory(history, model_path, data_name)

    # evaluate the model
    print("Evaluating the model")
    predictions = model.predict(X_test, batch_size=BATCHSIZE)
    predictions = np.argmax(predictions, axis=1)

    # get the classification report
    class_report = classification_report(y_test.argmax(axis=1), predictions, target_names=labels)
    print(class_report)

    # get the confusion matrix
    cf_matrix = confusion_matrix(y_test.argmax(axis=1), predictions)
    print(cf_matrix)
    
    np.save(os.path.join(model_path, data_name+'-cf.npy'), cf_matrix)
    np.save(os.path.join(model_path, data_name+'-class-report.npy'), class_report)



if __name__ == "__main__":
    main()