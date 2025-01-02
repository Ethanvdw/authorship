import re
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from collections import Counter
import numpy as np
import os
import joblib # more suitable for sklearn models


nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    """Basic text preprocessing: Lowercasing, punctuation removal, and stopword removal."""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = ' '.join(word for word in text.split() if word not in stop_words)
    return text

def read_file_content(file_path):
    """Reads the content of a text file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return None

def gather_text_data(base_folder):
    """Gathers text data from specified folders for each author."""
    texts = []
    authors = []

    for author_folder in os.listdir(base_folder):
         author_path = os.path.join(base_folder, author_folder)
         if os.path.isdir(author_path):
            for filename in os.listdir(author_path):
                if filename.endswith(".txt"):
                     file_path = os.path.join(author_path,filename)
                     text = read_file_content(file_path)
                     if text:
                        texts.append(text)
                        authors.append(author_folder)
    return texts, authors

def create_pairs(texts, authors):
    """Create pairs of texts with labels indicating same/different author."""
    pairs = []
    labels = []

    # Create pairs of same author
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            if authors[i] == authors[j]:
                pairs.append((texts[i], texts[j]))
                labels.append(1) # Same author
            else:
                pairs.append((texts[i], texts[j]))
                labels.append(0) # Different authors
    return pairs, labels



def extract_tfidf_features(text_pairs, vectorizer=None):
    """Extract TF-IDF features from text pairs."""
    corpus = []
    for text1, text2 in text_pairs:
      corpus.append(text1)
      corpus.append(text2)

    if vectorizer is None:
        vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(2,5))
        tfidf_matrix = vectorizer.fit_transform(corpus)
    else:
        tfidf_matrix = vectorizer.transform(corpus)

    # The first half of the matrix corresponds to text1 of the pairs, the second to text2
    num_pairs = len(text_pairs)
    feature_vectors = np.zeros((num_pairs, tfidf_matrix.shape[1])) 
    for i in range(num_pairs):
        vector1 = tfidf_matrix.toarray()[2*i]
        vector2 = tfidf_matrix.toarray()[2*i + 1]

        # Compute absolute difference between vectors
        feature_vectors[i] = np.abs(vector1 - vector2)
    return feature_vectors, vectorizer

def train_model(features, labels):
    """Train a logistic regression model."""
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42
    )

    # Calculate class weights to deal with imbalanced dataset.
    class_counts = Counter(y_train)
    class_weights = {0: len(y_train) / class_counts[0], 1 : len(y_train) / class_counts[1] }


    model = LogisticRegression(random_state=42, class_weight=class_weights, solver='liblinear')
    model.fit(X_train, y_train)
    return model, X_test, y_test

def save_model(model, vectorizer, model_path = "authorship_model.joblib"):
    """Saves model and vectorizer to disk."""
    model_data = {
        "model" : model,
        "vectorizer" : vectorizer
    }
    joblib.dump(model_data, model_path)
    print("Model saved to:", model_path)

def load_model(model_path = "authorship_model.joblib"):
     """Loads a model from disk."""
     try:
         model_data = joblib.load(model_path)
         model = model_data["model"]
         vectorizer = model_data["vectorizer"]
         print("Model loaded from:", model_path)
         return model, vectorizer
     except Exception as e:
        print(f"Error loading model {model_path}: {e}")
        return None, None
def evaluate_model(model, X_test, y_test):
    """Evaluate model performance on test data."""
    predictions = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, predictions))
    print("\nClassification Report:\n", classification_report(y_test, predictions))


def predict_authorship(model, text1, text2, vectorizer):
   
    # Preprocess both texts
    processed_text1 = preprocess_text(text1)
    processed_text2 = preprocess_text(text2)
    
    new_pairs = [(processed_text1, processed_text2)]

    new_features, _ = extract_tfidf_features(new_pairs, vectorizer=vectorizer)
    prediction = model.predict(new_features)[0]
    
    return prediction == 1 # True for same author, False otherwise

# Main script
if __name__ == "__main__":
    model_path = "authorship_model.joblib"
    
    # Ensure that there are folders for the texts
    base_folder = "texts"
    if not os.path.isdir(base_folder):
        print(f"Error: Please place the text files in folders within '{base_folder}'. E.g. '{base_folder}/Hemingway' and '{base_folder}/McCarthy'")
    else:
        # Check if we have a saved model
        model, vectorizer = load_model(model_path)

        if model is None: # If no model is found, create a new one
            texts, authors = gather_text_data(base_folder)
            processed_texts = [preprocess_text(text) for text in texts]
            pairs, labels = create_pairs(processed_texts, authors)
            tfidf_features, vectorizer = extract_tfidf_features(pairs)

            model, X_test, y_test = train_model(tfidf_features, labels)
            save_model(model, vectorizer, model_path)

            print("\nModel Evaluation:")
            evaluate_model(model, X_test, y_test)


        # Example prediction:
        text1 = "She was a woman of good sense and possessed a lively and cheerful temperament." # From Austen
        text2 = "Who controls the past controls the future. Who controls the present controls the past." # From Austen
        same_author = predict_authorship(model, text1, text2, vectorizer=vectorizer)
        if same_author:
            print("\nPredicted: These texts are from the same author.")
        else:
            print("\nPredicted: These texts are from different authors.")
