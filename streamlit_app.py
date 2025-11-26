import os
import requests
import streamlit as st

# ================================
# CONFIG
# ================================
BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")

st.set_page_config(
    page_title="Zentro – Intelligent Content Management",
    page_icon="🧠",
    layout="wide",
)

# ================================
# HEADER
# ================================
st.title("🧠 Zentro – Intelligent Content Management")
st.markdown(
    """
### Welcome to **Zentro**
An intelligent, local-first platform for:

- 🖼️ Image understanding (**Zentro Vision**)  
- 📄 Document analysis (**Zentro Docs**)  
- 💬 Conversational document Q&A (**Zentro Chat**)  
"""
)

# ================================
# SIDEBAR
# ================================
with st.sidebar:
    st.header("⚙️ Zentro Backend")
    st.text_input("Backend URL", value=BACKEND_URL, key="backend_url")

    if st.button("Check health"):
        try:
            r = requests.get(st.session_state.backend_url + "/health", timeout=5)
            st.success(r.json())
        except Exception as e:
            st.error(f"Health check failed: {e}")

# ================================
# TABS
# ================================
tab_image, tab_doc, tab_chat = st.tabs(
    ["🖼️ Zentro Vision", "📄 Zentro Docs", "💬 Zentro Chat"]
)

# ==============================================================================
# IMAGE TAB – Zentro Vision
# ==============================================================================
with tab_image:
    st.subheader("🖼️ Zentro Vision – Image Intelligence")

    img_instruction = st.text_area(
        "Instruction for Zentro Vision:",
        value="Describe this image in detail and extract visible text, objects, and key insights.",
        height=120,
    )

    img_file = st.file_uploader(
        "Upload an image",
        type=["png", "jpg", "jpeg", "webp"],
        key="image_uploader",
    )

    if img_file is not None:
        st.image(img_file, caption=img_file.name, use_container_width=True)

        if st.button("Analyze Image", type="primary"):
            try:
                files = {"file": (img_file.name, img_file.getvalue(), img_file.type)}
                data = {"instruction": img_instruction}
                with st.spinner("Zentro Vision analyzing image..."):
                    r = requests.post(
                        st.session_state.backend_url + "/analyze/image",
                        files=files,
                        data=data,
                        timeout=120,
                    )
                resp = r.json()

                if "error" in resp:
                    st.error(resp["error"])
                else:
                    st.subheader("🧠 Zentro Vision Result")
                    st.markdown(resp["result"])

            except Exception as e:
                st.error(f"Request failed: {e}")

# ==============================================================================
# DOCUMENT TAB – Zentro Docs
# ==============================================================================
with tab_doc:
    st.subheader("📄 Zentro Docs – Document Intelligence")

    doc_instruction = st.text_area(
        "Instruction for Zentro Docs:",
        value="Summarize this document and extract key points, entities, and important details.",
        height=120,
    )

    doc_file = st.file_uploader(
        "Upload a document",
        type=["pdf", "txt"],
        key="doc_uploader",
    )

    if doc_file is not None:
        st.write(f"Uploaded: **{doc_file.name}**")

        if st.button("Analyze Document", type="primary"):
            try:
                files = {"file": (doc_file.name, doc_file.getvalue(), doc_file.type)}
                data = {"instruction": doc_instruction}

                with st.spinner("Zentro Docs analyzing document..."):
                    r = requests.post(
                        st.session_state.backend_url + "/analyze/document",
                        files=files,
                        data=data,
                        timeout=240,
                    )

                resp = r.json()
                if "error" in resp:
                    st.error(resp["error"])
                else:
                    st.subheader("🧠 Zentro Docs Result")
                    st.markdown(resp["result"])

            except Exception as e:
                st.error(f"Request failed: {e}")

# ==============================================================================
# CHAT TAB – Zentro Chat (formerly RAG)
# ==============================================================================
with tab_chat:
    st.subheader("💬 Zentro Chat – Conversational Document AI")

    if "rag_doc_id" not in st.session_state:
        st.session_state.rag_doc_id = None
        st.session_state.rag_file_name = None
        st.session_state.rag_preview = None

    rag_file = st.file_uploader(
        "Upload a PDF or TXT for Zentro Chat",
        type=["pdf", "txt"],
        key="rag_uploader",
    )

    col1, col2 = st.columns([1, 2])

    with col1:
        if rag_file is not None:
            if st.button("Upload & Index Document", type="primary"):
                try:
                    files = {"file": (rag_file.name, rag_file.getvalue(), rag_file.type)}
                    with st.spinner("Indexing document for Zentro Chat..."):
                        r = requests.post(
                            st.session_state.backend_url + "/rag/upload",
                            files=files,
                            timeout=240,
                        )
                    resp = r.json()

                    if "error" in resp:
                        st.error(resp["error"])
                    else:
                        st.session_state.rag_doc_id = resp["doc_id"]
                        st.session_state.rag_file_name = resp.get("file_name", rag_file.name)
                        st.session_state.rag_preview = resp.get("preview", "")
                        st.success(f"Indexed document with ID: {resp['doc_id']}")

                except Exception as e:
                    st.error(f"Request failed: {e}")

    with col2:
        if st.session_state.rag_doc_id:
            st.info(
                f"Current indexed document: **{st.session_state.rag_file_name}** "
                f"(doc_id: `{st.session_state.rag_doc_id}`)"
            )

    if st.session_state.rag_preview:
        with st.expander("📄 Preview of extracted text", expanded=False):
            st.text(st.session_state.rag_preview)

    st.markdown("---")
    st.markdown("### 💬 Ask a question about the document")

    question = st.text_area(
        "Your question:",
        height=100,
    )

    rag_instruction = st.text_area(
        "Optional instruction for Zentro Chat:",
        value="Using only the given context, answer the question. If the answer is not in the document, say you don't know.",
        height=120,
    )

    if st.button("Ask Zentro Chat", type="primary"):
        if not st.session_state.rag_doc_id:
            st.warning("Upload and index a document first.")
        elif not question.strip():
            st.warning("Please type a question.")
        else:
            try:
                payload = {
                    "doc_id": st.session_state.rag_doc_id,
                    "question": question,
                    "instruction": rag_instruction,
                }

                with st.spinner("Zentro Chat analyzing..."):
                    r = requests.post(
                        st.session_state.backend_url + "/rag/ask",
                        json=payload,
                        timeout=240,
                    )

                resp = r.json()

                if "error" in resp:
                    st.error(resp["error"])
                else:
                    st.subheader("🧠 Zentro Chat Answer")
                    st.markdown(resp["answer"])
                    used_chunks = resp.get("used_chunks", [])
                    if used_chunks:
                        st.caption(f"Used chunk indices: {used_chunks}")

            except Exception as e:
                st.error(f"Request failed: {e}")
