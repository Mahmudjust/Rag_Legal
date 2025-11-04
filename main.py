import streamlit as st
import os
import tempfile
import subprocess
import requests
import numpy as np

from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import google.generativeai as genai

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
st.set_page_config(page_title="Fixed-PDF RAG (Bangla)", layout="centered")
st.title("Ask Anything About the Fixed Document")

# ---- Secrets (set in Streamlit → Settings → Secrets) ----
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
HF_TOKEN       = st.secrets.get("HF_TOKEN", "")   # optional, not used here

# ---- Fixed PDF (Google-Drive direct-download) ----
#   1. Make the file "Anyone with the link → Viewer"
#   2. Replace YOUR_FILE_ID with the ID from the share link
PDF_URL = "https://drive.google.com/uc?export=download&id=16VbhsygSSfHek-j31j_SkxSFdnEmrDEf"

# -------------------------------------------------
# 1. DOWNLOAD PDF (cached)
# -------------------------------------------------
@st.cache_data(show_spinner=False)
def download_pdf(url: str) -> str:
    """Download PDF from Google-Drive and return local path."""
    headers = {"User-Agent": "Mozilla/5.0"}
    r = requests.get(url, headers=headers, stream=True)
    if r.status_code != 200:
        st.error(f"Download failed (status {r.status_code}). Check the URL.")
        st.stop()

    # sanity-check
    if len(r.content) < 1000:
        st.error("File too small – probably got HTML instead of PDF.")
        st.stop()
    if not r.content.startswith(b"%PDF"):
        st.error("Downloaded file is NOT a PDF. Use the direct-download URL.")
        st.stop()

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    tmp.write(r.content)
    tmp.close()
    return tmp.name


# -------------------------------------------------
# 2. PDF → TEXT (pdftotext)
# -------------------------------------------------
@st.cache_data(show_spinner="Extracting text …")
def pdf_to_text(pdf_path: str) -> str:
    """Run pdftotext (installed on the container) and return full text."""
    # poppler-utils is installed via apt-get in the container (see Dockerfile below)
    result = subprocess.run(
        ["pdftotext", "-layout", pdf_path, "-"],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout


# -------------------------------------------------
# 3. CHUNK + EMBED + FAISS (cached once)
# -------------------------------------------------
@st.cache_resource(show_spinner="Building index …")
def build_index(_pdf_path: str):
    # 3a – text
    full_text = pdf_to_text(_pdf_path)

    # 3b – chunk
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separators=["\n\n", "\n", "।", ".", "!", "?", " ", ""],
        keep_separator=True,
    )
    chunks = splitter.split_text(full_text)

    # 3c – embed
    embedder = SentenceTransformer("intfloat/multilingual-e5-large")
    vectors = embedder.encode(chunks, normalize_embeddings=True, show_progress_bar=False)

    # 3d – FAISS
    dim = vectors.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(vectors))

    return chunks, index, embedder


# -------------------------------------------------
# 4. RETRIEVAL + GEMINI ANSWER
# -------------------------------------------------
def answer_question(query: str, chunks, faiss_index, embedder):
    # embed query
    q_vec = embedder.encode([query], normalize_embeddings=True)

    # retrieve top-k
    k = 3
    D, I = faiss_index.search(np.array(q_vec), k)
    retrieved = [chunks[i] for i in I[0] if i != -1]

    # build prompt
    context = "\n\n".join(retrieved)
    prompt = f"""
Use **only** the following context to answer the question in Bangla.

Context:
{context}

Question: {query}
Answer:
"""

    # call Gemini
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel("gemini-1.5-flash")   # works with 2.5-flash too
    response = model.generate_content(prompt)
    return response.text, retrieved


# -------------------------------------------------
# MAIN APP
# -------------------------------------------------
pdf_path = download_pdf(PDF_URL)
chunks, faiss_index, embedder = build_index(pdf_path)

st.success("Document loaded – ask anything in Bangla!")

question = st.text_input("Your question:", placeholder="উদাহরণ: মহাপরিচালক কীভাবে তথ্য-উপাত্ত ব্লক করতে পারেন?")

if question:
    with st.spinner("Retrieving & generating answer …"):
        answer, sources = answer_question(question, chunks, faiss_index, embedder)

    st.markdown("### Answer")
    st.write(answer)

    with st.expander("Sources (top 3 chunks)"):
        for i, src in enumerate(sources, 1):
            st.markdown(f"**Chunk {i}:**")
            st.code(src[:500] + ("…" if len(src) > 500 else ""), language="text")
