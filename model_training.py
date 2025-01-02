from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from collections import Counter
import joblib

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
