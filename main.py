from data_handling import gather_text_data
from preprocessing import preprocess_text
from feature_extraction import create_pairs, extract_tfidf_features
from model_training import train_model, save_model, load_model, evaluate_model
from prediction import predict_authorship
import os
import sys

# Main script
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python main.py <text1> <text2>")
        sys.exit(1)

    text1 = sys.argv[1]
    text2 = sys.argv[2]

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
        same_author = predict_authorship(model, text1, text2, vectorizer=vectorizer)
        if same_author:
            print("\nPredicted: These texts are from the same author.")
        else:
            print("\nPredicted: These texts are from different authors.")
