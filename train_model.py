import pandas as pd
import joblib

dataset = pd.read_csv('preprocessed_dataset.csv')

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

# Split dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(dataset['preprocessed_text'], dataset['sentiment'], test_size=0.2, random_state=42)

# Fill missing values in the 'preprocessed_text' column with an empty string
X_train = X_train.fillna('')

# TF-IDF vectorization
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)

# Save the TF-IDF vectorizer to a file
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')


# Train an SVM classifier
classifier = SVC(kernel='linear')
classifier.fit(X_train_tfidf, y_train)

# Save the trained model to a file
joblib.dump(classifier, 'sentiment_model.pkl')


# Evaluate the model
X_test = X_test.fillna('')
X_test_tfidf = vectorizer.transform(X_test)
accuracy = classifier.score(X_test_tfidf, y_test)
# print(f'Accuracy: {accuracy}')  
