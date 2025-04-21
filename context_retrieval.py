import re
import json
import sys
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Ensure nltk punkt is downloaded
nltk.download('punkt')

# ----------------------------
# Load lore data
# ----------------------------
with open("wiki_lore_dataset.json", "r", encoding="utf-8") as f:
    raw_data = json.load(f)

# ----------------------------
# Split content into smaller paragraphs
# ----------------------------
def chunk_text(entry):
    return nltk.sent_tokenize(entry["content"])

chunked_lore = []
for entry in raw_data:
    chunks = chunk_text(entry)
    chunked_lore.extend(chunks)

# ----------------------------
# TF-IDF Vectorization
# ----------------------------
vectorizer = TfidfVectorizer(stop_words="english", max_df=0.9)
tfidf_matrix = vectorizer.fit_transform(chunked_lore)

# ----------------------------
# Find best context using TF-IDF
# ----------------------------
def find_best_context_tfidf(question, top_k=3):
    question_vec = vectorizer.transform([question])
    similarities = cosine_similarity(question_vec, tfidf_matrix).flatten()
    top_indices = similarities.argsort()[-top_k:][::-1]
    top_contexts = [chunked_lore[i] for i in top_indices]
    return " ".join(top_contexts)

# ----------------------------
# Test block
# ----------------------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("❗ Usage: python context_retrieval.py \"Your question here\"")
        sys.exit(1)

    question_input = sys.argv[1]
    result = find_best_context_tfidf(question_input)
    print("\n🔍 Best Retrieved Context:")
    print(result if result else "⚠️ No relevant context found.")
