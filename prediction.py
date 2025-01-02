from preprocessing import preprocess_text
from feature_extraction import extract_tfidf_features

def predict_authorship(model, text1, text2, vectorizer):
    # Preprocess both texts
    processed_text1 = preprocess_text(text1)
    processed_text2 = preprocess_text(text2)
    
    new_pairs = [(processed_text1, processed_text2)]

    new_features, _ = extract_tfidf_features(new_pairs, vectorizer=vectorizer)
    prediction = model.predict(new_features)[0]
    
    return prediction == 1 # True for same author, False otherwise
