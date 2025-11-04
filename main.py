import streamlit as st
import requests
import tempfile
import numpy as np

from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import google.generativeai as genai
import fitz 


st.set_page_config(page_title="RAG Legal", layout="centered")
st.title("Ask About the Legal Document")


GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]


PDF_URL = "https://drive.google.com/uc?export=download&id=16VbhsygSSfHek-j31j_SkxSFdnEmrDEf"


@st.cache_data(show_spinner=False)
def download_pdf(url: str) -> str:
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers)
    if response.status_code != 200 or not response.content.startswith(b"%PDF"):
        st.error("Invalid PDF. Use Google Drive direct link: `uc?export=download&id=...`")
        st.stop()
    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    tmp_file.write(response.content)
    tmp_file.close()
    return tmp_file.name


@st.cache_data(show_spinner="Extracting text with PyMuPDF...")
def pdf_to_text(pdf_path: str) -> str:
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text("text")  
    doc.close()
    return text


@st.cache_resource(show_spinner="Building FAISS index...")
def build_index():
    pdf_path = download_pdf(PDF_URL)
    full_text = pdf_to_text(pdf_path)

    
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_text(full_text)

    
    model = SentenceTransformer("intfloat/multilingual-e5-large")
    vectors = model.encode(chunks, normalize_embeddings=True)

    
    dimension = vectors.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(vectors))

    return chunks, index, model


def answer_query(query: str, chunks, index, model):
    q_vec = model.encode([query], normalize_embeddings=True)
    D, I = index.search(np.array(q_vec), k=3)
    context = "\n\n".join([chunks[i] for i in I[0] if i != -1])

    prompt = f"""Use only the following context to answer in Bangla:

{context}

Question: {query}
Answer:"""

    genai.configure(api_key=GEMINI_API_KEY)
    response = genai.GenerativeModel("models/gemini-2.5-flash").generate_content(prompt)
    return response.text


try:
    chunks, index, model = build_index()
    st.success("Document loaded! Ask in Bangla.")
except Exception as e:
    st.error(f"Failed to load document: {e}")
    st.stop()

query = st.text_input("প্রশ্ন লিখুন:")
if query:
    with st.spinner("উত্তর তৈরি হচ্ছে..."):
        answer = answer_query(query, chunks, index, model)
    st.markdown("**উত্তর:**")
    st.write(answer)
