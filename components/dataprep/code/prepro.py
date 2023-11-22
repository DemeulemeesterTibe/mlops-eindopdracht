import pandas as pd
import nltk
import re
import os
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import argparse

def main():
    """ Main function of the script 
        It has the following steps:
            1. Read the dataset
            2. Drop the tweet_id column
            3. Drop the duplicated rows
            4. Normalize the text
        It has the following arguments:
            None    
    """
    defaultData = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'dataset', 'emotions.csv')
    defaultOutput = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'dataset', 'emotions-prepro.csv')
    parser = argparse.ArgumentParser(description='Preprocess the dataset')
    parser.add_argument('--data', type=str, default=defaultData, help='Path to the dataset')
    parser.add_argument('--output_path', type=str, default=defaultOutput, help='Path to the output data')
    args = parser.parse_args()

    output_path = args.output_path
    data = args.data

    nltk.download('stopwords')
    nltk.download('wordnet')
    stop_words = set(stopwords.words("english"))
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

    df = pd.read_csv(data)

    # drop the tweet_id column
    df.drop('tweet_id', axis=1, inplace=True)
    index = df[df['content'].duplicated() == True].index
    df.drop(index, axis = 0, inplace = True)
    df.reset_index(inplace=True, drop = True)
    df = normalize_text(df)
    df.to_csv(output_path, index=False)

if __name__ == '__main__':
    main()