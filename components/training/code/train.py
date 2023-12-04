import os
import argparse
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences



def main():
    parser = argparse.ArgumentParser(description='Preprocess the dataset')
    parser.add_argument('--training_data', type=str, help='Path to the training dataset folder')
    parser.add_argument('--testing_data', type=str, help='Path to the training dataset folder')
    parser.add_argument('--data_name', type=str, help='Name of the dataset')
    parser.add_argument('--epochs', type=str, help='Amount of the epochs for training')
    parser.add_argument('--output_folder', type=int, help='Output folder for the model')
    args = parser.parse_args()

    training_data = args.training_data
    testing_data = args.testing_data
    data_name = args.data_name + '-prepro'
    epochs = args.epochs
    output_folder = args.output_folder

    # read the training data
    training_df = pd.read_csv(os.path.join(training_data, data_name+'-training.csv'))
    
    # read the testing data
    testing_df = pd.read_csv(os.path.join(testing_data, data_name+'-testing.csv'))



    # encode the labels
    le = LabelEncoder()
    le.fit(training_df['label'])


if __name__ == "__main__":
    main()