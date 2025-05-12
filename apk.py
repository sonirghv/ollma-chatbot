import json
import re
import os
from pathlib import Path
import pickle
from typing import List, Dict, Any
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from pydantic import BaseModel
import pdfplumber
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from starlette.responses import JSONResponse
import requests
import torch
import openai
from dotenv import load_dotenv
load_dotenv()

app = FastAPI(title="Extractive PDF Chatbot Backend")

# --- Config ---
INDEX_DIR = "./index"
MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
START_PAGE = 3 
OLLAMA_URL = "http://localhost:11434/api/generate"
LLM_MODEL = "qwen3:1.7b"

# --- Global State ---
model = None
faiss_index = None
section_metadata = []
bm25_corpus = []
bm25 = None
chat_history = {}  # Dictionary to store chat history: {session_id: [{"query": str, "answer": str}, ...]}
# --- CORS Configuration ---
from fastapi.middleware.cors import CORSMiddleware

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# --- Models ---
class CompanyInfo(BaseModel):
    company_name: str
    company_location: str

class QueryRequest(CompanyInfo):
    query: str
    session_id: str
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
def get_company_folder(company_name: str, company_location: str) -> str:
    """Generate company-specific folder name"""
    return f"{company_name.lower()}_{company_location.lower()}"

def parse_toc(pdf_text: str) -> List[Dict[str, Any]]:
    """
    Parse the Table of Contents from the PDF text, specifically from pages 3 to 5.
    Matches entries in the format 'SectionNumber. Title PageNumber' (e.g., '1-1. Introduction 6').
    """
    toc = []
    toc_section = False
    section_pattern = re.compile(r'(\d+-\d+)\.\s+([^\n]+?)\s+(\d+)$')
    
    lines = pdf_text.split('\n')
    for line in lines:
        if line.strip() == "Table of Contents":
            toc_section = True
            continue
        if not toc_section or line.strip() in ['<CONTENT_FROM_OCR>', '</CONTENT_FROM_OCR>', '</PAGE>', '</DOCUMENT>'] or line.startswith('<DOCUMENT filename=') or line.startswith('<PAGE'):
            continue
        match = section_pattern.match(line.strip())
        if match:
            section_number = match.group(1)
            title = match.group(2).strip()
            page = int(match.group(3))
            toc.append({
                "section": section_number,
                "title": title,
                "page": page
            })
        if line.startswith('<PAGE6>'):
            break
    
    return toc

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

def save_toc(toc: List[Dict[str, Any]], filename: str, company_index_dir: str) -> None:
    """
    Save the table of contents to a text file as a single line.
    Each entry is formatted as 'SectionNumber: Title (Page PageNumber)' and separated by ' | '.
    """
    toc_entries = [
        f"{section['section']}: {section['title']} (Page {section['page']})"
        for section in toc
    ]
    toc_line = " | ".join(toc_entries)
    toc_file_path = os.path.join(company_index_dir, f"{Path(filename).stem}_toc.txt")
    os.makedirs(company_index_dir, exist_ok=True)
    with open(toc_file_path, 'w', encoding='utf-8') as f:
        f.write(toc_line)

def process_pdf(file: UploadFile, filename: str, company_index_dir: str) -> None:
    """Process and index a PDF file"""
    try:
        pdf_text = ""
        
        # Read PDF
        with pdfplumber.open(file.file) as pdf:
            for page in pdf.pages:
                if page.page_number < START_PAGE and page.page_number > 5:
                    continue
                page_text = page.extract_text() or ""
                pdf_text += f"<PAGE{page.page_number}>\n<CONTENT_FROM_OCR>\n{page_text}\n</CONTENT_FROM_OCR>\n</PAGE{page.page_number}>\n"

        # Extract TOC
        toc = parse_toc(pdf_text)
        save_toc(toc, filename, company_index_dir)

        # Parse sections
        sections = parse_pdf_content(pdf_text)
        filename_stem = Path(filename).stem

        # Process sections for indexing
        for section in sections:
            text = f"{section['title']}\n\n{section['content']}"
            words = text.split()
            word_count = len(words)

            if word_count <= 300:
                # Process as single chunk for sections <= 300 words
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
            else:
                # Chunk sections > 300 words with 70-word overlap
                chunk_size = 300
                overlap = 70
                for i in range(0, word_count, chunk_size - overlap):
                    chunk_words = words[i:i + chunk_size]
                    if len(chunk_words) < 50:
                        continue
                    chunk_text = " ".join(chunk_words)
                    embedding = model.encode(chunk_text, normalize_embeddings=True, batch_size=32)
                    faiss_index.add(np.array([embedding]).astype('float32'))
                    section_metadata.append({
                        'filename': filename_stem,
                        'section': f"{section['section']}_chunk_{i//(chunk_size-overlap)+1}",
                        'title': section['title'],
                        'page': section['page'],
                        'content': chunk_text
                    })
                    tokenized_text = chunk_text.lower().split()
                    bm25_corpus.append(tokenized_text)

        # Update BM25
        bm25 = BM25Okapi(bm25_corpus)

        # Save indices
        os.makedirs(company_index_dir, exist_ok=True)
        faiss.write_index(faiss_index, os.path.join(company_index_dir, 'faiss_index.bin'))
        with open(os.path.join(company_index_dir, 'metadata.pkl'), 'wb') as f:
            pickle.dump(section_metadata, f)
        with open(os.path.join(company_index_dir, 'bm25_corpus.pkl'), 'wb') as f:
            pickle.dump(bm25_corpus, f)

    except Exception as e:
        # Clean up any partial files if processing fails
        if os.path.exists(company_index_dir):
            for file in os.listdir(company_index_dir):
                if file.endswith(('.bin', '.pkl')):
                    os.remove(os.path.join(company_index_dir, file))
        raise Exception(f"Error processing PDF: {str(e)}")

def hybrid_search(query: str, top_k: int, company_folder: str = None) -> List[Dict[str, Any]]:
    """Perform hybrid search using both FAISS and BM25"""
    # Load indices for the specific company if provided
    if company_folder:
        company_index_dir = os.path.join(INDEX_DIR, company_folder)
        if os.path.exists(os.path.join(company_index_dir, 'faiss_index.bin')):
            faiss_index = faiss.read_index(os.path.join(company_index_dir, 'faiss_index.bin'))
        if os.path.exists(os.path.join(company_index_dir, 'metadata.pkl')):
            with open(os.path.join(company_index_dir, 'metadata.pkl'), 'rb') as f:
                section_metadata.extend(pickle.load(f))
        if os.path.exists(os.path.join(company_index_dir, 'bm25_corpus.pkl')):
            with open(os.path.join(company_index_dir, 'bm25_corpus.pkl'), 'rb') as f:
                bm25_corpus.extend(pickle.load(f))
            bm25 = BM25Okapi(bm25_corpus)

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
        combined_results[idx] = combined_results.get(idx, 0) + 0.85 * score  
    for idx, score in zip(bm25_top_indices, bm25_top_scores):
        combined_results[idx] = combined_results.get(idx, 0) + 0.15 * (score / max(bm25_scores) if max(bm25_scores) > 0 else 0)

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

client = openai.OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.environ.get("GROQ_API_KEY")
)

def generate_llm_answer(query: str, context_chunks: List[Dict[str, Any]], session_id: str, company_folder: str = None) -> str:
    """Generate an answer using the LLM with context and chat history"""
    # Construct context from top-2 chunks
    context = ""
    for i, chunk in enumerate(context_chunks[:5], 1):
        context += f"Chunk {i} (Page {chunk['page']}, Section {chunk['section']}):\n{chunk['text']}\n\n"

    # Retrieve and format chat history
    history_context = ""
    if session_id in chat_history:
        # Limit to last 5 interactions to manage token limits
        recent_history = chat_history[session_id][-5:]
        for i, interaction in enumerate(recent_history, 1):
            history_context += f"Interaction {i}:\nUser Query: {interaction['query']}\nAssistant Answer: {interaction['answer']}\n\n"

    # Read company-specific handbook TOC
    toc_path = os.path.join(INDEX_DIR, company_folder, f"{company_folder}_toc.txt") if company_folder else None
    toc_text = ""
    if toc_path and os.path.exists(toc_path):
        with open(toc_path, 'r', encoding='utf-8') as f:
            toc_text = f.read()

    # Construct prompt
    prompt = f"""You are a helpful assistant specializing in HR policy documents for a company. Your goal is to provide accurate, comprehensive, and user-friendly answers based on the provided context and Table of Contents (TOC). Follow these guidelines:

1. **Answering Specific Questions**:
   - Use verbatim or near-verbatim text from the context for the specific topic asked, paying attention to synonyms, similar phrases, and the semantic intent of the question.
   - Include **all relevant information** from **all provided context chunks**, covering scenarios, conditions, exceptions, prerequisites, and related details.
   - Synthesize information across chunks to provide a cohesive answer, avoiding repetition unless necessary for clarity.
   - Cite the **page number** and **section** for each piece of information **only once** at the end of the final answer, in the format: "(Page X, Section Y)".
   - If the context lacks relevant information, respond with: "Sorry, I am not able to find the exact information. Can you please rephrase your question or specify what you mean? For example, are you asking about [list 2-3 related topics from the 'Table of context' or context]?"

2. **Handling General Queries**:
   - For greetings (e.g., "Hello", "Hi"), respond politely: "Hello! How can I assist you with the HR Manual today?"
   - For expressions of gratitude (e.g., "Thank you", "Thanks"), respond: "You're welcome! Do you have any other questions about the HR Manual?"
   - For vague or unclear queries, respond with: "Can you clarify what you mean by '{query}'? I have information on [list 2-3 related topics from the 'Table of context' or context]. Please specify or choose one of these options."

3. **Thought Process**:
   - Use <think> tags to explain your reasoning, such as how you interpreted the query, which context chunks are relevant, and why.
   - **Do not include** chunk details (e.g., page numbers, section numbers, or the word "chunk") in the <think> section. Refer to context generally (e.g., "the provided HR policy text").
   - Keep the thought process concise and focused on query interpretation and context relevance.

4. **Response Structure**:
   - Wrap the thought process in <think>...</think> tags.
   - Provide the final answer outside the <think> tags, ensuring it is clear, complete, and includes all relevant details.
   - End the final answer with citations in the format: "(Page X, Section Y)" for each piece of information, listed only once.

5. **Using the Table of Contents**:
   - Refer to the 'Table of context' to identify relevant sections when the query is broad or unclear.
   - Use the 'Table of context' to suggest related topics when asking for clarification.

**Table of Contents**:
{toc_text}

**companyname and location**:
{company_folder}

**Chat History** (previous interactions for context):
{history_context}

**Current Question**:
{query}

**Document Context**:
{context}
"""

    res = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": query}
        ],
        temperature=0.0,
        max_tokens=32768,
        timeout=30
    )
    answer = res.choices[0].message.content

    # Store the query and answer in chat history
    if session_id not in chat_history:
        chat_history[session_id] = []
    chat_history[session_id].append({"query": query, "answer": answer})

    return answer

# --- Startup Event ---
@app.on_event("startup")
def startup_event():
    global model, faiss_index, section_metadata, bm25_corpus, bm25
    
    model = SentenceTransformer(MODEL_NAME, device='cuda' if torch.cuda.is_available() else 'cpu')
    dimension = model.get_sentence_embedding_dimension()
    faiss_index = faiss.IndexFlatIP(dimension)

# --- Endpoints ---
@app.post("/upload_pdf", response_model=dict)
async def upload_pdf(
    file: UploadFile = File(...),
    company_name: str = Form(...),
    company_location: str = Form(...)
):
    """Upload and process a PDF file with company information"""
    try:
        # Validate file type
        if not file.filename.endswith(('.pdf', '.xml')):
            raise HTTPException(
                status_code=400,
                detail="Only PDF or XML files are supported"
            )

        # Validate company information
        if not company_name.strip() or not company_location.strip():
            raise HTTPException(
                status_code=400,
                detail="Company name and location are required"
            )

        # Create company-specific folder
        company_folder = get_company_folder(company_name, company_location)
        company_index_dir = os.path.join(INDEX_DIR, company_folder)
        os.makedirs(company_index_dir, exist_ok=True)

        # Process PDF with company-specific folder
        try:
            process_pdf(file, file.filename, company_index_dir)
            return {
                "message": f"Document {file.filename} indexed successfully for {company_name} ({company_location})",
                "company_folder": company_folder
            }
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Error processing document: {str(e)}"
            )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error: {str(e)}"
        )

@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """Process a query and return a response"""
    try:
        # Get company-specific folder
        company_folder = get_company_folder(request.company_name, request.company_location)
        company_index_dir = os.path.join(INDEX_DIR, company_folder)

        # Check if company folder exists
        if not os.path.exists(company_index_dir):
            raise HTTPException(
                status_code=404,
                detail=f"No documents found for {request.company_name} ({request.company_location})"
            )

        # Fetch top-2 chunks for context
        results = hybrid_search(
            request.query,
            top_k=2,
            company_folder=company_folder
        )

        if not results:
            # Store the query and response in chat history even if no results
            chat_history[request.session_id] = chat_history.get(request.session_id, [])
            chat_history[request.session_id].append({
                "query": request.query,
                "answer": f"There is no relevant information in the documents for {request.company_name} ({request.company_location}). Please consult with management directly."
            })
            return QueryResponse(
                answer=f"There is no relevant information in the documents for {request.company_name} ({request.company_location}). Please consult with management directly.",
                page=None,
                section=None,
                score=None
            )

        # Use LLM for answer generation with chat history
        answer = generate_llm_answer(
            request.query,
            results,
            request.session_id,
            company_folder
        )

        # Return response with metadata from top result
        top_result = results[0]
        return QueryResponse(
            answer=answer,
            page=top_result['page'],
            section=top_result['section'],
            score=top_result['score']
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing query: {str(e)}"
        )

@app.delete("/clear_history/{session_id}", response_model=dict)
async def clear_history(session_id: str):
    """Clear chat history for a specific session"""
    try:
        if session_id in chat_history:
            del chat_history[session_id]
            return {"message": f"Chat history for session {session_id} cleared successfully"}
        return {"message": f"No chat history found for session {session_id}"}
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error clearing chat history: {str(e)}"
        )

@app.get("/context/{query}", response_model=List[ContextChunk])
async def get_context(
    query: str,
    company_name: str,
    company_location: str,
    top_k: int = 1
):
    """Get context chunks for a query"""
    try:
        company_folder = get_company_folder(company_name, company_location)
        results = hybrid_search(
            query,
            top_k,
            company_folder
        )
        return [
            ContextChunk(
                text=result['text'],
                page=result['page'],
                section=result['section'],
                score=result['score']
            ) for result in results
        ]
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving context: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)