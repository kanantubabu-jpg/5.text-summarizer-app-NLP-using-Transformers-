import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
# 1. Modern Deep Learning Summarizer (Abstractive)
def abstractive_summarize(text):
    if not text.strip():
        return "Please enter some text to summarize."
    
    try:
        # Import transformers lazily so module import doesn't fail when
        # heavy dependencies (transformers/torch) are not installed.
        try:
            from transformers import pipeline
        except Exception:
            return None
        max_len = min(150, int(input_length * 0.5))
        min_len = min(30, int(input_length * 0.2))
        
        if max_len <= min_len:
            max_len = min_len + 10

        summary = summarizer(text, max_length=max_len, min_length=min_len, do_sample=False)
        return summary[0]['summary_text']
    except Exception as e:
        return f"Error in Deep Learning Summary: {str(e)}"


# 2. Traditional Machine Learning Summarizer (Extractive using Scikit-Learn & NumPy)
def extractive_summarize(text, num_sentences=2):
    if not text.strip():
        return "Please enter some text to summarize."
        
    # Split the text into basic sentences
    sentences = [s.strip() for s in text.split('.') if s.strip()]
    if len(sentences) <= num_sentences:
        return text

    # Compute TF-IDF Vectors using Scikit-Learn
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(sentences)
    
    # Sum the TF-IDF scores for each sentence using NumPy
    sentence_scores = np.array(tfidf_matrix.sum(axis=1)).flatten()
    
    # Get indices of the top highest-scoring sentences
    top_indices = np.argsort(sentence_scores)[-num_sentences:]
    top_indices.sort()  # Keep them in original order of appearance
    
    summary = " ".join([sentences[i] + "." for i in top_indices])
    return summary