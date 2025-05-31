import pandas as pd
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer


dataset = pd.read_csv('training_dataset.csv',header=None)

def preprocess_text(text):
    
    if isinstance(text, str):
        words = word_tokenize(text)
        words = [word.lower() for word in words if word.isalpha()]
        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        words = [word for word in words if word not in stop_words]
        # Stemming
        stemmer = PorterStemmer()
        words = [stemmer.stem(word) for word in words]
        return ' '.join(words)
    else:
        # Handle non-string values 
        return ''


# Apply preprocessing to the 'text' column
dataset['preprocessed_text'] = dataset[3].apply(preprocess_text)

dataset = dataset[dataset[2] != 'Irrelevant']

# Saving data into preprocessed_dataset.csv file
dataset.to_csv('preprocessed_dataset.csv', header=["id", "entity", "sentiment", "text", "preprocessed_text"], index=False)
