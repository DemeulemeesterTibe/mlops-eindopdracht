import pandas as pd
import nltk
import re
import os
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
import argparse
def main():
    """ Main function of the script 
        It has the following steps:
            1. Read the dataset
            2. Drop the tweet_id column
            3. Drop the duplicated rows
            4. Normalize the text
            5. Save the preprocessed data
        It has the following arguments:
            --data: Path to the dataset
            --output_path: Path to the output data
            --language: Language of the dataset
    """
    # set the default values
    defaultData = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'dataset', 'emotions.csv')
    defaultOutput = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'dataset', 'emotions-prepro.csv')
    defaultLang = 'english'

    # get the arguments
    parser = argparse.ArgumentParser(description='Preprocess the dataset')
    parser.add_argument('--data', type=str, default=defaultData, help='Path to the dataset folder')
    parser.add_argument('--data_name', type=str, default=defaultData, help='Name of the dataset')
    parser.add_argument('--output_path', type=str, default=defaultOutput, help='Path to the output data')
    parser.add_argument('--language', type=str, default=defaultLang, help='Language of the dataset')
    args = parser.parse_args()

    data_name = args.data_name
    language = args.language

    # download the required packages
    nltk.download('stopwords')
    nltk.download('wordnet')
    stop_words = set(stopwords.words(language))
    def lemmatization(text):
        lemmatizer= WordNetLemmatizer()

        text = text.split()

        text=[lemmatizer.lemmatize(y) for y in text]
        
        return " " .join(text)

    def remove_stop_words(text):

        Text=[i for i in str(text).split() if i not in stop_words]
        return " ".join(Text)

    def removing_numbers(text):
        text=''.join([i for i in text if not i.isdigit()])
        return text

    def lower_case(text):
        
        text = text.split()

        text=[y.lower() for y in text]
        
        return " " .join(text)

    def removing_punctuations(text):
        ## Remove punctuations
        text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,،-./:;<=>؟?@[\]^_`{|}~"""), ' ', text)
        text = text.replace('؛',"", )
        
        ## remove extra whitespace
        text = re.sub('\s+', ' ', text)
        text =  " ".join(text.split())
        return text.strip()

    def removing_urls(text):
        url_pattern = re.compile(r'https?://\S+|www\.\S+')
        return url_pattern.sub(r'', text)
                
    def normalize_text(df):
        df.content=df.content.apply(lambda text : lower_case(text))
        df.content=df.content.apply(lambda text : remove_stop_words(text))
        df.content=df.content.apply(lambda text : removing_numbers(text))
        df.content=df.content.apply(lambda text : removing_punctuations(text))
        df.content=df.content.apply(lambda text : removing_urls(text))
        df.content=df.content.apply(lambda text : lemmatization(text))
        return df

    # read the dataset
    print("Reading the dataset")
    
    df = pd.read_csv(os.path.join(args.data, data_name+'.csv'))

    # drop the tweet_id column
    df.drop('tweet_id', axis=1, inplace=True)

    # drop the duplicated rows
    index = df[df['content'].duplicated() == True].index
    df.drop(index, axis = 0, inplace = True)
    df.reset_index(inplace=True, drop = True)

    # normalize the text
    df = normalize_text(df)

    # encode the labels
    encoder = LabelEncoder()
    df['sentiment'] = encoder.fit_transform(df['sentiment'])
    
    # save the preprocessed data
    print("Saving the preprocessed csv file")
    df.to_csv(os.path.join(args.output_path, data_name+'-prepro.csv'), index=False)

if __name__ == '__main__':
    main()