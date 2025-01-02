import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

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
