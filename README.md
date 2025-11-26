# 🚀 Zentro – Intelligent Content Management
### *Open-Source AI for Image, Document & Conversational Intelligence*
### *Powered by Cyclops-VL 2.0 & Optimized for NVIDIA RTX GPUs*

Zentro is an open-source intelligent content platform built for advanced image analysis, document understanding, and retrieval-augmented conversational intelligence (RAG).
It operates fully offline on your machine and is optimized for NVIDIA RTX GPUs.

> ⚡ A commercial **cloud-managed version** also exists, where all computation, processing, indexing, storage, and orchestration run entirely in the cloud.

---

## ⚡ GPU Optimization (Open-Source Version)

- CUDA-accelerated inference
- Mixed Precision (AMP)
- TensorRT-friendly model architecture
- GPU-accelerated embedding generation
- Optimized for RTX 2050 → 4090 & A-series  

---

## ✨ Key Features

### 🖼️ Zentro Vision – Image Intelligence
- Object, text, and layout detection
- Diagram & UI screenshot analysis
- Image summarization & reasoning

### 📄 Zentro Docs – Document Intelligence
- PDF/TXT parsing
- Metadata extraction
- Structured summaries, entities & topics

### 💬 Zentro Chat – RAG-Powered Conversational AI
Includes a complete offline Retrieval-Augmented Generation system:

- Document ingestion
- Automatic chunking
- GPU-accelerated embeddings
- Cosine similarity retrieval
- Context-grounded answers using Cyclops-VL 2.0

---

## 🧠 Zentro RAG Architecture

1. **Ingestion** → PDF/TXT loading
2. **Chunking** → Optimized text segmentation
3. **Embedding** → SentenceTransformers (GPU accelerated)
4. **Retrieval** → Cosine-similarity search
5. **Answering** → Cyclops-VL 2.0 generates grounded responses

---

## 🧩 Technology Stack

| Component | Technology |
|----------|------------|
| UI | Streamlit |
| Backend | FastAPI |
| Vision-Language Model | Cyclops-VL 2.0 |
| GPU Acceleration | CUDA + RTX |
| Embeddings | SentenceTransformers |
| Retrieval | Custom cosine similarity |
| Parsing | PyPDF + text utilities |
| API Format | OpenAI-compatible |
| Environment | Conda |

---

## 📦 Project Structure

```
zentro/
│── backend.py
│── streamlit_app.py
│── environment.yml
│── README.md
```

---

# 🌐 Commercial Cloud Version (Optional)

Alongside the offline open-source version, Zentro is also available as a **fully cloud-managed commercial offering**.

In the cloud version:

- All computation
- All data processing
- All document analysis
- All retrieval, indexing, and similarity search
- All orchestration and automation
- All storage and management  

…are handled entirely **in the cloud**, requiring no local hardware.

Extra cloud capabilities include:

- Multi-user workspace
- Document ingestion pipelines
- OCR + handwriting recognition
- Knowledge graph generation
- Centralized storage
- Role-based access control
- Audit logs and monitoring dashboards  

> 📌 This README describes the offline open-source version.  
> The cloud-managed version is a separate commercial product.

---

## 🔒 Privacy (Open-Source Version)

- Fully offline  
- No telemetry
- No external APIs  
- All data stays on-device

---

## 🚀 Running Zentro (Offline Version)

### Start Backend
```
uvicorn backend:app --reload --host 127.0.0.1 --port 8000
```

Health check:
```
http://127.0.0.1:8000/health
```

### Start Frontend
```
streamlit run streamlit_app.py
```

UI:
```
http://localhost:8501
```

Backend URL:
```
http://127.0.0.1:8000
```

---

## 🛠 Troubleshooting

### Backend not running
```
uvicorn backend:app --reload
```

### Chat not responding
Document not indexed or backend offline.

### GPU not detected
```
import torch
torch.cuda.is_available()
```

---

## 🤝 Contributing

Contributions are welcome!

---

## 📜 License  
MIT License

---

## ⭐ Support  
If Zentro helps you, star ⭐ the repo!
