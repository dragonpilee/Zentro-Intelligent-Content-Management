import base64
import io
import os
import uuid
from typing import Dict, List, Optional

import numpy as np
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from pydantic import BaseModel
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer

# =========================
# CONFIG
# =========================

# LM Studio server (OpenAI-compatible)
LM_STUDIO_BASE_URL = os.getenv("LM_STUDIO_BASE_URL", "http://127.0.0.1:1234/v1")
LM_STUDIO_API_KEY = os.getenv("LM_STUDIO_API_KEY", "lm-studio")  # dummy key, LM Studio ignores it

# Model names as shown in LM Studio
QWEN_VL_MODEL_NAME = os.getenv("QWEN_VL_MODEL_NAME", "qwen/qwen3-vl-4b-instruct")
QWEN_CHAT_MODEL_NAME = os.getenv("QWEN_CHAT_MODEL_NAME", QWEN_VL_MODEL_NAME)

client = OpenAI(
    base_url=LM_STUDIO_BASE_URL,
    api_key=LM_STUDIO_API_KEY,
)

# Small, fast embedding model
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# In-memory RAG store: doc_id -> {chunks, embeddings}
RAG_STORE: Dict[str, Dict[str, object]] = {}

app = FastAPI(title="Qwen Backend")

# Allow Streamlit on localhost
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================
# HELPERS
# =========================

def encode_image_to_data_url(file_bytes: bytes, mime_type: str) -> str:
    b64 = base64.b64encode(file_bytes).decode("utf-8")
    return f"data:{mime_type};base64,{b64}"


def extract_text_from_pdf(file_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(file_bytes))
    pages_text: List[str] = []
    for page in reader.pages:
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        if text.strip():
            pages_text.append(text)
    return "\n\n".join(pages_text)


def guess_mime_type(file_name: str) -> str:
    f = file_name.lower()
    if f.endswith(".png"):
        return "image/png"
    if f.endswith(".jpg") or f.endswith(".jpeg"):
        return "image/jpeg"
    if f.endswith(".webp"):
        return "image/webp"
    return "application/octet-stream"


def chunk_text(text: str, max_chars: int = 800) -> List[str]:
    """Simple chunker by characters with paragraph splitting."""
    paragraphs = text.split("\n\n")
    chunks: List[str] = []
    current = ""

    for para in paragraphs:
        if len(current) + len(para) + 2 <= max_chars:
            current += ("\n\n" + para) if current else para
        else:
            if current:
                chunks.append(current)
            current = para

    if current:
        chunks.append(current)

    return chunks


def cosine_similarities(query_vec: np.ndarray, doc_matrix: np.ndarray) -> np.ndarray:
    query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-8)
    doc_norm = doc_matrix / (np.linalg.norm(doc_matrix, axis=1, keepdims=True) + 1e-8)
    return doc_norm @ query_norm


def call_qwen_chat(messages, model_name: Optional[str] = None, temperature: float = 0.2) -> str:
    model_name = model_name or QWEN_CHAT_MODEL_NAME
    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=temperature,
    )
    return response.choices[0].message.content


# =========================
# SCHEMAS
# =========================

class RAGQuestion(BaseModel):
    doc_id: str
    question: str
    instruction: Optional[str] = None


# =========================
# ROUTES
# =========================

@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/analyze/image")
async def analyze_image(
    file: UploadFile = File(...),
    instruction: str = Form("Describe this image in detail."),
):
    file_bytes = await file.read()
    mime_type = guess_mime_type(file.filename)

    data_url = encode_image_to_data_url(file_bytes, mime_type)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": instruction},
                {"type": "image_url", "image_url": {"url": data_url}},
            ],
        }
    ]

    try:
        result = call_qwen_chat(messages, model_name=QWEN_VL_MODEL_NAME)
        return {"result": result}
    except Exception as e:
        return {"error": str(e)}


@app.post("/analyze/document")
async def analyze_document(
    file: UploadFile = File(...),
    instruction: str = Form("Summarize this document and extract key points, entities, and dates."),
):
    file_bytes = await file.read()
    name = file.filename.lower()

    if name.endswith(".txt"):
        try:
            text = file_bytes.decode("utf-8", errors="ignore")
        except Exception:
            text = file_bytes.decode("latin-1", errors="ignore")
    elif name.endswith(".pdf"):
        text = extract_text_from_pdf(file_bytes)
    else:
        return {"error": "Unsupported file type. Use PDF or TXT."}

    if not text.strip():
        return {"error": "No text extracted from document."}

    truncated = text[:8000]

    prompt = (
        f"You are an AI assistant analyzing a document.\n\n"
        f"USER INSTRUCTION:\n{instruction}\n\n"
        f"DOCUMENT CONTENT (possibly truncated):\n{truncated}"
    )

    messages = [{"role": "user", "content": prompt}]

    try:
        result = call_qwen_chat(messages)
        return {"result": result}
    except Exception as e:
        return {"error": str(e)}


@app.post("/rag/upload")
async def rag_upload(
    file: UploadFile = File(...),
):
    file_bytes = await file.read()
    name = file.filename.lower()

    if name.endswith(".txt"):
        try:
            text = file_bytes.decode("utf-8", errors="ignore")
        except Exception:
            text = file_bytes.decode("latin-1", errors="ignore")
    elif name.endswith(".pdf"):
        text = extract_text_from_pdf(file_bytes)
    else:
        return {"error": "Unsupported file type. Use PDF or TXT."}

    if not text.strip():
        return {"error": "No text extracted from document."}

    chunks = chunk_text(text, max_chars=800)
    embeddings = embedding_model.encode(chunks)  # shape: (n_chunks, dim)

    doc_id = str(uuid.uuid4())
    RAG_STORE[doc_id] = {
        "chunks": chunks,
        "embeddings": embeddings,
        "file_name": file.filename,
    }

    preview = text[:1000]

    return {
        "doc_id": doc_id,
        "num_chunks": len(chunks),
        "file_name": file.filename,
        "preview": preview,
    }


@app.post("/rag/ask")
async def rag_ask(body: RAGQuestion):
    if body.doc_id not in RAG_STORE:
        return {"error": "Unknown doc_id. Upload the document again."}

    data = RAG_STORE[body.doc_id]
    chunks: List[str] = data["chunks"]
    embeddings: np.ndarray = data["embeddings"]

    q_vec = embedding_model.encode([body.question])[0]
    sims = cosine_similarities(q_vec, embeddings)

    top_k = min(5, len(chunks))
    top_indices = np.argsort(-sims)[:top_k]

    context_parts = []
    for idx in top_indices:
        context_parts.append(f"[Chunk {idx}]\n{chunks[idx]}")
    context_text = "\n\n".join(context_parts)

    instruction = body.instruction or (
        "Using only the context chunks below, answer the user's question. "
        "If the answer is not clearly in the context, say you don't know."
    )

    prompt = (
        f"{instruction}\n\n"
        f"CONTEXT CHUNKS:\n{context_text}\n\n"
        f"USER QUESTION:\n{body.question}"
    )

    messages = [{"role": "user", "content": prompt}]

    try:
        answer = call_qwen_chat(messages)
        return {
            "answer": answer,
            "used_chunks": [int(i) for i in top_indices.tolist()],
        }
    except Exception as e:
        return {"error": str(e)}
