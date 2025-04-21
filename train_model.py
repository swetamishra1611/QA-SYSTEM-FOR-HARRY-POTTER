from transformers import T5ForConditionalGeneration, T5Tokenizer, Trainer, TrainingArguments, DataCollatorForSeq2Seq
from datasets import Dataset
import pandas as pd
import json
import re
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split

# ----------------------------
# Load QA data
# ----------------------------
with open("qa_dataset.json", "r", encoding="utf-8") as f:
    qa_data = json.load(f)

# Ensure it's a list of entries
if isinstance(qa_data, dict):
    qa_data = [qa_data]

df = pd.DataFrame(qa_data)

# Normalize column names
if "answer_html" in df.columns:
    df.rename(columns={"answer_html": "answer"}, inplace=True)

assert "question" in df.columns and "answer" in df.columns, "Missing required fields"

# ----------------------------
# Load Lore Data
# ----------------------------
with open("wiki_lore_dataset.json", "r", encoding="utf-8") as f:
    lore_data = json.load(f)

contexts = [entry["content"] for entry in lore_data]
vectorizer = TfidfVectorizer(stop_words='english')
context_vectors = vectorizer.fit_transform(contexts)

# ----------------------------
# Define context retrieval
# ----------------------------
def find_best_context(question: str) -> str:
    question_vector = vectorizer.transform([question])
    similarity_scores = cosine_similarity(question_vector, context_vectors)
    best_index = similarity_scores.argmax()
    return contexts[best_index]

# ----------------------------
# Add context to dataset
# ----------------------------
print("🔍 Adding context to questions...")

df["retrieved_context"] = df["question"].apply(find_best_context)
df["input_text"] = df.apply(lambda row:
    f"You are an expert on Harry Potter lore. Based on the following context, answer the question accurately.\n"
    f"Context: {row['retrieved_context']}\nQuestion: {row['question']}", axis=1)
df["target_text"] = df["answer"]

# ----------------------------
# Train/Validation split
# ----------------------------
train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)
train_dataset = Dataset.from_pandas(train_df[["input_text", "target_text"]].reset_index(drop=True))
val_dataset = Dataset.from_pandas(val_df[["input_text", "target_text"]].reset_index(drop=True))

# ----------------------------
# Load Model and Tokenizer
# ----------------------------
model_name = "t5-small"  # Or use "t5-base"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# ----------------------------
# Preprocessing
# ----------------------------
def preprocess(example):
    model_inputs = tokenizer(
        example["input_text"],
        max_length=512,
        padding="max_length",
        truncation=True,
    )
    labels = tokenizer(
        example["target_text"],
        max_length=128,
        padding="max_length",
        truncation=True
    )
    labels["input_ids"] = [(token if token != tokenizer.pad_token_id else -100) for token in labels["input_ids"]]
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_train_dataset = train_dataset.map(preprocess, remove_columns=train_dataset.column_names)
tokenized_val_dataset = val_dataset.map(preprocess, remove_columns=val_dataset.column_names)

# ----------------------------
# Training setup
# ----------------------------
training_args = TrainingArguments(
    output_dir="./qa_model",
    per_device_train_batch_size=4,
    num_train_epochs=5,
    learning_rate=3e-5,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_steps=10,
    report_to="none"  # disable WandB or others
)

data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_val_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator
)

# ----------------------------
# Train
# ----------------------------
print("🚀 Starting training...")
trainer.train()

# ----------------------------
# Save Model
# ----------------------------
print("💾 Saving model to 'final_model'...")
model.save_pretrained("final_model")
tokenizer.save_pretrained("final_model")
print("✅ Model saved.")
