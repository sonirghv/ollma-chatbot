import json
import re
import os
from pathlib import Path
import pickle
from typing import List, Dict, Any
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import pdfplumber
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from starlette.responses import JSONResponse
import requests
import torch

app = FastAPI(title="Extractive PDF Chatbot Backend")

# --- Config ---
INDEX_DIR = "./index"
MODEL_NAME = "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"
START_PAGE = 6  # Skip cover pages, TOC, etc.
OLLAMA_URL = "http://localhost:11434/api/generate"
LLM_MODEL = "deepseek-r1:1.5b"

# --- Global State ---
model = None
faiss_index = None
section_metadata = []
bm25_corpus = []
bm25 = None

# --- Models ---
class QueryRequest(BaseModel):
    query: str
    top_k: int = 1

class QueryResponse(BaseModel):
    answer: str
    page: int | None = None
    section: str | None = None
    score: float | None = None

class ContextChunk(BaseModel):
    text: str
    page: int
    section: str
    score: float

# --- Helper Functions ---
def parse_pdf_content(pdf_text: str) -> List[Dict[str, Any]]:
    sections = []
    current_section = None
    current_content = []
    current_page = 1
    section_pattern = re.compile(r'(\d+-\d+)\.\s+([^\n]+)')

    lines = pdf_text.split('\n')
    for line in lines:
        if line.startswith('<PAGE'):
            page_num = int(re.search(r'\d+', line).group())
            current_page = page_num
            continue
        if line.strip() in ['<CONTENT_FROM_OCR>', '</CONTENT_FROM_OCR>', '</PAGE>', '</DOCUMENT>'] or line.startswith('<DOCUMENT filename='):
            continue
        section_match = section_pattern.match(line.strip())
        if section_match:
            if current_section:
                sections.append({
                    "section": current_section["number"],
                    "title": current_section["title"],
                    "page": current_section["page"],
                    "content": "\n".join(current_content).strip()
                })
                current_content = []
            section_number = section_match.group(1)
            section_title = section_match.group(2).strip()
            current_section = {
                "number": section_number,
                "title": section_title,
                "page": current_page
            }
        else:
            if current_section:
                current_content.append(line.strip())
    if current_section and current_content:
        sections.append({
            "section": current_section["number"],
            "title": current_section["title"],
            "page": current_section["page"],
            "content": "\n".join(current_content).strip()
        })
    return sections

def process_pdf(file: UploadFile, filename: str) -> None:
    global faiss_index, section_metadata, bm25_corpus, bm25
    pdf_text = ""
    
    # Read PDF
    with pdfplumber.open(file.file) as pdf:
        for page in pdf.pages:
            if page.page_number < START_PAGE:
                continue
            page_text = page.extract_text() or ""
            pdf_text += f"<PAGE{page.page_number}>\n<CONTENT_FROM_OCR>\n{page_text}\n</CONTENT_FROM_OCR>\n</PAGE{page.page_number}>\n"

    # Parse sections
    sections = parse_pdf_content(pdf_text)
    filename_stem = Path(filename).stem

    # Process sections
    for section in sections:
        text = f"{section['title']}\n\n{section['content']}"
        embedding = model.encode(text, normalize_embeddings=True, batch_size=32)
        faiss_index.add(np.array([embedding]).astype('float32'))
        section_metadata.append({
            'filename': filename_stem,
            'section': section['section'],
            'title': section['title'],
            'page': section['page'],
            'content': section['content']
        })
        tokenized_text = text.lower().split()
        bm25_corpus.append(tokenized_text)

    # Update BM25
    bm25 = BM25Okapi(bm25_corpus)

    # Save indices
    os.makedirs(INDEX_DIR, exist_ok=True)
    faiss.write_index(faiss_index, os.path.join(INDEX_DIR, 'faiss_index.bin'))
    with open(os.path.join(INDEX_DIR, 'metadata.pkl'), 'wb') as f:
        pickle.dump(section_metadata, f)
    with open(os.path.join(INDEX_DIR, 'bm25_corpus.pkl'), 'wb') as f:
        pickle.dump(bm25_corpus, f)

def hybrid_search(query: str, top_k: int) -> List[Dict[str, Any]]:
    if not section_metadata:
        return []

    # Generate query embedding
    query_embedding = model.encode(query, normalize_embeddings=True)
    query_embedding = np.array([query_embedding]).astype('float32')

    # FAISS search
    distances, indices = faiss_index.search(query_embedding, top_k)
    faiss_scores = 1 - distances[0]  

    # BM25 search
    tokenized_query = query.lower().split()
    bm25_scores = bm25.get_scores(tokenized_query)
    bm25_top_indices = np.argsort(bm25_scores)[::-1][:top_k]
    bm25_top_scores = bm25_scores[bm25_top_indices]

    # Combine scores
    combined_results = {}
    for idx, score in zip(indices[0], faiss_scores):
        combined_results[idx] = combined_results.get(idx, 0) + 0.7 * score
    for idx, score in zip(bm25_top_indices, bm25_top_scores):
        combined_results[idx] = combined_results.get(idx, 0) + 0.3 * (score / max(bm25_scores) if max(bm25_scores) > 0 else 0)

    # Sort results
    sorted_results = sorted(combined_results.items(), key=lambda x: x[1], reverse=True)[:top_k]
    results = []
    for idx, score in sorted_results:
        section = section_metadata[idx]
        results.append({
            'text': f"{section['title']}\n\n{section['content']}",
            'page': section['page'],
            'section': section['section'],
            'score': score
        })
    return results

def generate_llm_answer(query: str, context_chunks: List[Dict[str, Any]]) -> str:
    # Construct context from top-2 chunks
    context = ""
    for i, chunk in enumerate(context_chunks[:2], 1):
        context += f"Chunk {i} (Page {chunk['page']}, Section {chunk['section']}):\n{chunk['text']}\n\n"

    # Construct prompt
    prompt = f"""You are a helpful assistant. When answering the user's question, strictly extract the answer directly from the provided context. Use verbatim or near-verbatim text for the specific topic asked—do not paraphrase, summarize, or condense the information.
    Include every piece of information from the context that is relevant to the question, covering all possible scenarios, conditions, exceptions, and related details. If there are any conditions, prerequisites, or special cases mentioned in the context, ensure they are included in your answer.
    For each piece of information you provide, cite the page number and section (if available) from the context.
    If the context does not contain relevant information, respond with:
    “There is no relevant information in the 'HR Manual'. Please consult with management directly.”
    Do not provide any information that is not explicitly present in the context.

**Question**: {query}

**Context**:
{context}
"""

    # Call Ollama API
    payload = {
        "model": LLM_MODEL,
        "prompt": prompt,
        "stream": False,
        
    }
    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=120)
        response.raise_for_status()
        return response.json().get("response", "Error: No response from LLM")
    except Exception as e:
        return f"Error calling LLM: {str(e)}"

# --- Startup Event ---
@app.on_event("startup")
def startup_event():
    global model, faiss_index, section_metadata, bm25_corpus, bm25
    
    model = SentenceTransformer(MODEL_NAME, device='cuda' if torch.cuda.is_available() else 'cpu')
    dimension = model.get_sentence_embedding_dimension()
    faiss_index = faiss.IndexFlatIP(dimension)  # Inner Product for cosine similarity

    # Load existing indices if available
    if os.path.exists(os.path.join(INDEX_DIR, 'faiss_index.bin')):
        faiss_index = faiss.read_index(os.path.join(INDEX_DIR, 'faiss_index.bin'))
    if os.path.exists(os.path.join(INDEX_DIR, 'metadata.pkl')):
        with open(os.path.join(INDEX_DIR, 'metadata.pkl'), 'rb') as f:
            section_metadata.extend(pickle.load(f))
    if os.path.exists(os.path.join(INDEX_DIR, 'bm25_corpus.pkl')):
        with open(os.path.join(INDEX_DIR, 'bm25_corpus.pkl'), 'rb') as f:
            bm25_corpus.extend(pickle.load(f))
        bm25 = BM25Okapi(bm25_corpus)

# --- Endpoints ---
@app.post("/upload_pdf", response_model=dict)
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(('.pdf', '.xml')):
        raise HTTPException(status_code=400, detail="Only PDF or XML files are supported")
    
    try:
        process_pdf(file, file.filename)
        return {"message": f"Document {file.filename} indexed successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")

@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    # Fetch top-2 chunks for context (in case LLM is needed)
    results = hybrid_search(request.query, top_k=2)
    if not results:
        return QueryResponse(
            answer="There is no relevant information in the 'HR Manual'. Please consult with management directly.",
            page=None,
            section=None,
            score=None
        )
    
    # Check word count of top chunk
    top_result = results[0]
    word_count = len(top_result['text'].split())
    
    if word_count < 140:
        # Return chunk directly if < 100 words
        answer = top_result['text']
    else:
        # Use LLM for answer if >= 100 words
        answer = generate_llm_answer(request.query, results)
    
    # Return response with metadata from top result
    return QueryResponse(
        answer=answer,
        page=top_result['page'],
        section=top_result['section'],
        score=top_result['score']
    )

@app.get("/context/{query}", response_model=List[ContextChunk])
async def get_context(query: str, top_k: int = 1):
    results = hybrid_search(query, top_k)
    return [
        ContextChunk(
            text=result['text'],
            page=result['page'],
            section=result['section'],
            score=result['score']
        ) for result in results
    ]

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)