README.txt — Fictional Universe QA System (Harry Potter Lore)
--------------------------------------------------------------

📚 Project Name:
Fictional Universe QA System (Harry Potter Lore)

🛠 Technologies Used:
- Python 3.8+
- FastAPI
- Transformers (Hugging Face)
- Datasets (Hugging Face)
- scikit-learn
- Uvicorn
- Pandas
- Jinja2 Templates
- HTML/CSS/JS (Frontend)

📁 Directory Structure:
project-root/
├── main.py                      # FastAPI backend
├── context_retrieval.py        # TF-IDF based context retriever
├── templates/
│   └── index.html              # Frontend HTML
├── qa_dataset.json             # JSON file of QA pairs
├── wiki_lore_dataset.json      # JSON file of fictional universe context
├── train_model.py              # Script to train T5 model
├── final_model/                # Folder containing saved fine-tuned model
│   ├── config.json
│   ├── pytorch_model.bin
│   ├── tokenizer_config.json
│   └── ...
└── README.txt                  # This file

⚙️ Setup Instructions:
1. Clone or download this repository.

2. Create a virtual environment:
   python -m venv venv
   source venv/bin/activate  # or `venv\Scripts\activate` on Windows

3. Install required packages:
   pip install -r requirements.txt

4. Train the T5 model (optional):
   If you want to fine-tune the model from scratch, run:
   python train_model.py

   This script will:
   - Load QA and lore data
   - Retrieve best matching context using TF-IDF
   - Create `input_text` and `target_text` for T5
   - Fine-tune T5 on this dataset
   - Save the model in `final_model/`

5. Run the FastAPI server:
   python main.py

6. Access the Web Interface:
   Open your browser and go to:
   http://127.0.0.2:8002/

🧠 How It Works:
1. User submits a lore-related question (e.g., “What are Horcruxes?”).
2. The FastAPI backend uses `context_retrieval.py` to fetch the top context from `wiki_lore_dataset.json` using TF-IDF.
3. The `final_model` (fine-tuned T5) uses the question + context as input.
4. The model generates an accurate answer based on the provided lore.
5. The answer is returned and displayed on the web page.

📦 Sample Question:
Question: Why did Voldemort create Horcruxes?

Expected Answer (model-generated):
To achieve immortality by splitting his soul and hiding it in multiple objects.

✅ requirements.txt Example:
fastapi
uvicorn
jinja2
transformers
scikit-learn
pandas
datasets
transformers>=4.40.0
datasets>=2.18.0
pandas>=2.0.0
beautifulsoup4>=4.12.0
requests>=2.31.0
scikit-learn>=1.3.0
torch>=2.2.0
tqdm>=4.66.0

✨ Acknowledgments:
- Hugging Face for the T5 model and `datasets` library.
- The Harry Potter Fandom Wiki (for context data).
- Inspired by real-world open-domain QA systems.
