from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from transformers import T5Tokenizer, T5ForConditionalGeneration
import uvicorn
from context_retrieval import find_best_context_tfidf

# Initialize FastAPI app
app = FastAPI()

# Set up templates directory
templates = Jinja2Templates(directory="templates")

# Load the fine-tuned T5 model and tokenizer
tokenizer = T5Tokenizer.from_pretrained("final_model")
model = T5ForConditionalGeneration.from_pretrained("final_model")

# Request schema for incoming JSON
class QAInput(BaseModel):
    question: str

# Web page route (renders the form)
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# API route for question-answering
@app.post("/ask")
async def ask_question(payload: QAInput):
    try:
        question = payload.question.strip()

        if not question:
            return JSONResponse(content={"answer": "Please enter a valid question."}, status_code=400)

        # Get relevant context using TF-IDF retrieval
        context = find_best_context_tfidf(question, top_k=2)

        # Format input for T5 model
        input_text = f"context: {context} question: {question}"
        input_ids = tokenizer.encode(input_text, return_tensors="pt", truncation=True, max_length=512)

        # Generate answer
        output_ids = model.generate(input_ids, max_length=128)
        answer = tokenizer.decode(output_ids[0], skip_special_tokens=True)

        return {
            "question": question,
            "answer": answer,
            "used_context": context[:300] + "..."  # Optional: context preview
        }

    except Exception as e:
        return JSONResponse(
            content={"answer": f"An error occurred: {str(e)}"},
            status_code=500
        )

# Run FastAPI app
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.2", port=8002)
