import pandas as pd
import numpy as np
import argparse
import os

def main():

    parser = argparse.ArgumentParser(description='Preprocess the dataset')
    parser.add_argument('--data', type=str, help='Path to the dataset folder')
    parser.add_argument('--data_name', type=str, help='Name of the dataset')
    parser.add_argument('--training_data_output', type=str, help='Path to the training output data')
    parser.add_argument('--testing_data_output', type=str, help='Path to the testing output data')
    parser.add_argument('--split_size', type=int, help='Language of the dataset')
    args = parser.parse_args()

    data = args.data
    data_name = args.data_name + '-prepro'
    train_test_split = args.split_size

    df = pd.read_csv(os.path.join(data, data_name+'.csv'))


    # shuffle the data
    df = df.sample(frac=1).reset_index(drop=True)

    # split the data into train and test csv files
    msk = np.random.rand(len(df)) < train_test_split
    train = df[msk]
    test = df[~msk]

    # save the train and test dataframes to csv files
    train.to_csv(os.path.join(args.training_data_output, data_name+'-training.csv'), index=False)
    test.to_csv(os.path.join(args.testing_data_output, data_name+'-testing.csv'), index=False)


if __name__ == '__main__':
    main()