from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import shutil
import os

from rag.loader import load_pdf
from rag.embedder import get_embeddings
from rag.vectorstore import VectorStore
from agents.agent import agent_answer

app = FastAPI()

# ✅ Enable frontend connection
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow all (for development)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global vector store
vectorstore = None

# File path
UPLOAD_PATH = "data/sample.pdf"


# Request schema
class QueryRequest(BaseModel):
    query: str


# Health check
@app.get("/")
def home():
    return {"message": "🚀 RAG Backend is Running"}


# 📄 Upload PDF
@app.post("/upload/")
async def upload_pdf(file: UploadFile = File(...)):
    global vectorstore

    # Create data folder if not exists
    os.makedirs("data", exist_ok=True)

    # Save uploaded file
    with open(UPLOAD_PATH, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Load text
    texts = load_pdf(UPLOAD_PATH)

    # Generate embeddings
    embeddings = get_embeddings(texts)

    # Create FAISS vector store
    vectorstore = VectorStore(len(embeddings[0]))
    vectorstore.add(embeddings, texts)

    return {
        "message": "✅ PDF uploaded and processed successfully",
        "chunks": len(texts)
    }


# ❓ Ask Question
@app.post("/ask/")
def ask_question(req: QueryRequest):
    global vectorstore

    if vectorstore is None:
        return {"error": "❌ Please upload a PDF first"}

    query = req.query

    # Embed query
    query_embedding = get_embeddings([query])

    # Search relevant chunks
    results = vectorstore.search(query_embedding, k=3)

    # Build context
    context = "\n\n".join([r["text"] for r in results])

    # Generate answer
    answer = agent_answer(query, context)

    # Compute relevance score
    scores = [1 / (1 + r["distance"]) for r in results]
    relevance_score = round(max(scores), 3)

    return {
        "query": query,
        "answer": answer,
        "relevance_score": relevance_score,
        "top_chunks": [
            {
                "text": r["text"][:300],
                "score": round(1 / (1 + r["distance"]), 3)
            }
            for r in results
        ]
    }


