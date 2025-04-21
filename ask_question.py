from transformers import T5ForConditionalGeneration, T5Tokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
import json
import os
import re
import sys

nltk.download('punkt')

MODEL_PATH = "final_model"
USE_FALLBACK = False

# ----------------------------
# Check and load model/tokenizer
# ----------------------------
required_files = ["spiece.model", "tokenizer_config.json", "config.json", "pytorch_model.bin"]

if not os.path.exists(MODEL_PATH) or any(
    not os.path.exists(os.path.join(MODEL_PATH, f)) for f in required_files
):
    print(f"⚠️ Warning: Required files missing in '{MODEL_PATH}'. Falling back to 't5-small' model.")
    USE_FALLBACK = True
    MODEL_PATH = "t5-small"

try:
    tokenizer = T5Tokenizer.from_pretrained(MODEL_PATH)
    model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH)
except Exception as e:
    print(f"❌ Failed to load model/tokenizer from '{MODEL_PATH}': {e}")
    sys.exit(1)

# ----------------------------
# Load lore context and chunk
# ----------------------------
try:
    with open("wiki_lore_dataset.json", "r", encoding="utf-8") as f:
        raw_data = json.load(f)
except FileNotFoundError:
    print("❌ wiki_lore_dataset.json not found. Please ensure the lore file is available.")
    sys.exit(1)

# Split into smaller sentence chunks
def chunk_text(entry):
    return nltk.sent_tokenize(entry["content"])

chunked_lore = []
for entry in raw_data:
    chunked_lore.extend(chunk_text(entry))

# ----------------------------
# Prepare TF-IDF model
# ----------------------------
vectorizer = TfidfVectorizer(stop_words="english", max_df=0.9)
tfidf_matrix = vectorizer.fit_transform(chunked_lore)

def find_best_context(question: str, top_k=3) -> str:
    question_vec = vectorizer.transform([question])
    similarities = cosine_similarity(question_vec, tfidf_matrix).flatten()
    top_indices = similarities.argsort()[-top_k:][::-1]
    top_contexts = [chunked_lore[i] for i in top_indices]
    return " ".join(top_contexts)

# ----------------------------
# Ask question using auto context
# ----------------------------
def ask_question_with_auto_context(question):
    context = find_best_context(question)
    input_text = f"context: {context} question: {question}"
    input_ids = tokenizer(input_text, return_tensors="pt", truncation=True).input_ids
    outputs = model.generate(input_ids, max_length=128)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# ----------------------------
# Ask question using manual context
# ----------------------------
def ask_question_manual(question, context):
    input_text = f"context: {context} question: {question}"
    input_ids = tokenizer(input_text, return_tensors="pt", truncation=True).input_ids
    outputs = model.generate(input_ids, max_length=128)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# ----------------------------
# Save fallback model if needed
# ----------------------------
if USE_FALLBACK:
    print("\n💾 Saving fallback model to 'final_model' for future use...")
    os.makedirs("final_model", exist_ok=True)
    model.save_pretrained("final_model")
    tokenizer.save_pretrained("final_model")
    print("✅ Saved model and tokenizer to 'final_model'")

# ----------------------------
# Example usage
# ----------------------------
if __name__ == "__main__":
    question = "Why did Gilderoy try to mend Harry's arm?"

    # Option 1: Auto-retrieved context from lore
    answer_auto = ask_question_with_auto_context(question)
    print("\n🧠 Auto Context Answer:")
    print("Q:", question)
    print("A:", answer_auto)

    # Option 2: Manually provided context (optional)
    manual_context = (
        "During a Quidditch match, Harry Potter broke his arm. "
        "Gilderoy Lockhart, a boastful and incompetent professor, insisted on fixing it. "
        "Instead of healing the bone, he accidentally removed all the bones from Harry's arm."
    )
    answer_manual = ask_question_manual(question, manual_context)
    print("\n✍️ Manual Context Answer:")
    print("Q:", question)
    print("A:", answer_manual)
