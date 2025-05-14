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
import logging
from fastapi.middleware.cors import CORSMiddleware
from threading import Lock
from time import sleep
from requests.exceptions import HTTPError
import redis
import io

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('chatbot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

load_dotenv()

app = FastAPI(title="Extractive PDF Chatbot Backend")


# --- Config ---
INDEX_DIR = "./index"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
START_PAGE = 3 
OLLAMA_URL = "http://localhost:11434/api/generate"
LLM_MODEL = "qwen3:1.7b"
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_DB = int(os.getenv("REDIS_DB", 0))
CACHE_TTL = 3600  # Cache expiration in seconds (1 hour)

# --- Global State ---
model = None
chat_history = {}
company_locks = {}
redis_client = None

# --- CORS Configuration ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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
    """Generate company-specific folder name with sanitization"""
    clean_company_name = re.sub(r'[^\w\s-]', '_', company_name.lower().strip())
    clean_company_location = re.sub(r'[^\w\s-]', '_', company_location.lower().strip())
    company_folder = f"{clean_company_name}_{clean_company_location}"
    logger.info(f"Generated company folder: {company_folder}")
    return company_folder

def get_company_lock(company_folder: str) -> Lock:
    """Get or create a lock for a company folder"""
    if company_folder not in company_locks:
        company_locks[company_folder] = Lock()
    return company_locks[company_folder]

def parse_toc(pdf_text: str) -> List[Dict[str, Any]]:
    """Parse the Table of Contents from the PDF text"""
    logger.info("Parsing TOC")
    toc = []
    toc_section = False
    section_pattern = re.compile(r'(\d+-\d+)\.?\s+([^\n]+?)\s+(\d+)$')
    
    lines = pdf_text.split('\n')
    for line in lines:
        if line.strip() == "Table of Contents":
            toc_section = True
            logger.info("Found TOC section")
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
    
    logger.info(f"Parsed TOC with {len(toc)} entries: {toc}")
    return toc

def parse_pdf_content(pdf_text: str) -> List[Dict[str, Any]]:
    """Parse PDF content into sections"""
    logger.info("Parsing PDF content")
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
    
    logger.info(f"Parsed {len(sections)} sections")
    return sections

def save_toc(toc: List[Dict[str, Any]], filename: str, company_index_dir: str) -> None:
    """Save the table of contents to a text file"""
    logger.info(f"Saving TOC for {filename} to {company_index_dir}")
    toc_entries = [
        f"{section['section']}: {section['title']} (Page {section['page']})"
        for section in toc
    ]
    toc_line = " | ".join(toc_entries) if toc_entries else "No TOC entries found"
    toc_file_path = os.path.join(company_index_dir, f"{Path(filename).stem}_toc.txt")
    os.makedirs(company_index_dir, exist_ok=True)
    with open(toc_file_path, 'w', encoding='utf-8') as f:
        f.write(toc_line)
    logger.info(f"TOC saved to {toc_file_path}")

def process_pdf(file: UploadFile, filename: str, company_index_dir: str) -> None:
    """Process and index a PDF file"""
    logger.info(f"Processing PDF: {filename}, Company dir: {company_index_dir}")
    try:
        pdf_text = ""
        with pdfplumber.open(file.file) as pdf:
            for page in pdf.pages:
                if page.page_number < START_PAGE and page.page_number > 5:
                    continue
                page_text = page.extract_text() or ""
                pdf_text += f"<PAGE{page.page_number}>\n<CONTENT_FROM_OCR>\n{page_text}\n</CONTENT_FROM_OCR>\n</PAGE{page.page_number}>\n"
        logger.info(f"Extracted PDF text (length: {len(pdf_text)} characters)")

        # Extract TOC
        toc = parse_toc(pdf_text)
        save_toc(toc, filename, company_index_dir)

        # Initialize local state
        section_metadata = []
        bm25_corpus = []
        dimension = model.get_sentence_embedding_dimension()
        local_faiss_index = faiss.IndexFlatIP(dimension)
        logger.info("Initialized new FAISS index")

        # Parse sections
        sections = parse_pdf_content(pdf_text)
        filename_stem = Path(filename).stem

        # Batch process sections for embeddings
        texts = []
        section_info = []
        for section in sections:
            text = f"{section['title']}\n\n{section['content']}"
            words = text.split()
            word_count = len(words)
            logger.info(f"Processing section {section['section']} with {word_count} words")

            if word_count <= 300:
                texts.append(text)
                section_info.append({
                    'filename': filename_stem,
                    'section': section['section'],
                    'title': section['title'],
                    'page': section['page'],
                    'content': section['content']
                })
                tokenized_text = text.lower().split()
                bm25_corpus.append(tokenized_text)
            else:
                chunk_size = 300
                overlap = 70
                for i in range(0, word_count, chunk_size - overlap):
                    chunk_words = words[i:i + chunk_size]
                    if len(chunk_words) < 50:
                        continue
                    chunk_text = " ".join(chunk_words)
                    texts.append(chunk_text)
                    section_info.append({
                        'filename': filename_stem,
                        'section': f"{section['section']}_chunk_{i//(chunk_size-overlap)+1}",
                        'title': section['title'],
                        'page': section['page'],
                        'content': chunk_text
                    })
                    tokenized_text = chunk_text.lower().split()
                    bm25_corpus.append(tokenized_text)

        # Generate embeddings in batch
        if texts:
            logger.info(f"Generating embeddings for {len(texts)} texts")
            embeddings = model.encode(texts, normalize_embeddings=True, batch_size=32)
            for embedding, info in zip(embeddings, section_info):
                local_faiss_index.add(np.array([embedding]).astype('float32'))
                section_metadata.append(info)
            logger.info("Added embeddings to FAISS index")

        # Update BM25
        bm25 = BM25Okapi(bm25_corpus)
        logger.info("Updated BM25 index")

        # Save indices to disk
        logger.info(f"Saving files to {company_index_dir}")
        os.makedirs(company_index_dir, exist_ok=True)
        faiss_index_path = os.path.join(company_index_dir, 'faiss_index.bin')
        metadata_path = os.path.join(company_index_dir, 'metadata.pkl')
        bm25_corpus_path = os.path.join(company_index_dir, 'bm25_corpus.pkl')
        
        faiss.write_index(local_faiss_index, faiss_index_path)
        logger.info("Saved FAISS index")
        with open(metadata_path, 'wb') as f:
            pickle.dump(section_metadata, f)
        logger.info("Saved metadata.pkl")
        with open(bm25_corpus_path, 'wb') as f:
            pickle.dump(bm25_corpus, f)
        logger.info("Saved bm25_corpus.pkl")

        # Cache in Redis
        company_folder = os.path.basename(company_index_dir)
        if redis_client:
            try:
                # Cache FAISS index
                with open(faiss_index_path, 'rb') as f:
                    faiss_bytes = f.read()
                redis_client.setex(f"faiss:{company_folder}", CACHE_TTL, faiss_bytes)
                logger.info(f"Cached FAISS index in Redis for {company_folder}")

                # Cache metadata
                with open(metadata_path, 'rb') as f:
                    metadata_bytes = f.read()
                redis_client.setex(f"metadata:{company_folder}", CACHE_TTL, metadata_bytes)
                logger.info(f"Cached metadata in Redis for {company_folder}")

                # Cache BM25 corpus
                with open(bm25_corpus_path, 'rb') as f:
                    bm25_bytes = f.read()
                redis_client.setex(f"bm25_corpus:{company_folder}", CACHE_TTL, bm25_bytes)
                logger.info(f"Cached BM25 corpus in Redis for {company_folder}")
            except Exception as e:
                logger.error(f"Error caching in Redis: {str(e)}", exc_info=True)
        else:
            logger.warning("Redis client not available, skipping caching")
    except Exception as e:
        logger.error(f"Error processing PDF: {str(e)}", exc_info=True)
    #     if os.path.exists(company_index_dir):
    #         for file in os.listdir(company_index_dir):
    #             if file.endswith(('.bin', '.pkl', '.txt')):
    #                 try:
    #                     os.remove(os.path.join(company_index_dir, file))
    #                     logger.info(f"Cleaned up file: {file}")
    #                 except Exception as cleanup_err:
    #                     logger.error(f"Error cleaning up file {file}: {str(cleanup_err)}")
    #     raise Exception(f"Error processing PDF: {str(e)}")

def hybrid_search(query: str, top_k: int, company_folder: str = None) -> List[Dict[str, Any]]:
    """Perform hybrid search using both FAISS and BM25 with Redis caching"""
    logger.info(f"Performing hybrid search for query: {query}, company: {company_folder}")
    local_metadata = []
    local_bm25_corpus = []
    dimension = model.get_sentence_embedding_dimension()
    local_faiss_index = faiss.IndexFlatIP(dimension)

    if company_folder:
        company_index_dir = os.path.join(INDEX_DIR, company_folder)
        faiss_path = os.path.join(company_index_dir, 'faiss_index.bin')
        metadata_path = os.path.join(company_index_dir, 'metadata.pkl')
        bm25_path = os.path.join(company_index_dir, 'bm25_corpus.pkl')

        try:
            if redis_client:
                # Try loading from Redis
                faiss_key = f"faiss:{company_folder}"
                metadata_key = f"metadata:{company_folder}"
                bm25_key = f"bm25_corpus:{company_folder}"

                # Load FAISS index
                faiss_bytes = redis_client.get(faiss_key)
                if faiss_bytes:
                    logger.info(f"Loading FAISS index from Redis for {company_folder}")
                    faiss_io = io.BytesIO(faiss_bytes)
                    local_faiss_index = faiss.read_index(faiss_io)
                else:
                    logger.info(f"FAISS index not in Redis, loading from disk: {faiss_path}")
                    if os.path.exists(faiss_path):
                        local_faiss_index = faiss.read_index(faiss_path)
                        with open(faiss_path, 'rb') as f:
                            faiss_bytes = f.read()
                        redis_client.setex(faiss_key, CACHE_TTL, faiss_bytes)
                        logger.info(f"Cached FAISS index in Redis for {company_folder}")
                    else:
                        logger.warning(f"No FAISS index found at {faiss_path}")
                        return []

                # Load metadata
                metadata_bytes = redis_client.get(metadata_key)
                if metadata_bytes:
                    logger.info(f"Loading metadata from Redis for {company_folder}")
                    local_metadata = pickle.loads(metadata_bytes)
                else:
                    logger.info(f"Metadata not in Redis, loading from disk: {metadata_path}")
                    if os.path.exists(metadata_path):
                        with open(metadata_path, 'rb') as f:
                            local_metadata = pickle.load(f)
                        metadata_bytes = pickle.dumps(local_metadata)
                        redis_client.setex(metadata_key, CACHE_TTL, metadata_bytes)
                        logger.info(f"Cached metadata in Redis for {company_folder}")
                    else:
                        logger.warning(f"No metadata found at {metadata_path}")
                        return []

                # Load BM25 corpus
                bm25_bytes = redis_client.get(bm25_key)
                if bm25_bytes:
                    logger.info(f"Loading BM25 corpus from Redis for {company_folder}")
                    local_bm25_corpus = pickle.loads(bm25_bytes)
                else:
                    logger.info(f"BM25 corpus not in Redis, loading from disk: {bm25_path}")
                    if os.path.exists(bm25_path):
                        with open(bm25_path, 'rb') as f:
                            local_bm25_corpus = pickle.load(f)
                        bm25_bytes = pickle.dumps(local_bm25_corpus)
                        redis_client.setex(bm25_key, CACHE_TTL, bm25_bytes)
                        logger.info(f"Cached BM25 corpus in Redis for {company_folder}")
                    else:
                        logger.warning(f"No BM25 corpus found at {bm25_path}")
                        return []

                bm25 = BM25Okapi(local_bm25_corpus)
            else:
                logger.warning("Redis client not available, loading from disk")
                # Load from disk
                if os.path.exists(faiss_path):
                    local_faiss_index = faiss.read_index(faiss_path)
                    logger.info(f"Loaded FAISS index from disk: {faiss_path}")
                else:
                    logger.warning(f"No FAISS index found at {faiss_path}")
                    return []
                if os.path.exists(metadata_path):
                    with open(metadata_path, 'rb') as f:
                        local_metadata = pickle.load(f)
                    logger.info(f"Loaded metadata from disk: {metadata_path}")
                else:
                    logger.warning(f"No metadata found at {metadata_path}")
                    return []
                if os.path.exists(bm25_path):
                    with open(bm25_path, 'rb') as f:
                        local_bm25_corpus = pickle.load(f)
                    logger.info(f"Loaded BM25 corpus from disk: {bm25_path}")
                    bm25 = BM25Okapi(local_bm25_corpus)
                else:
                    logger.warning(f"No BM25 corpus found at {bm25_path}")
                    return []
        except Exception as e:
            logger.error(f"Error accessing Redis or disk: {str(e)}", exc_info=True)
            # Fallback to disk
            if os.path.exists(faiss_path):
                local_faiss_index = faiss.read_index(faiss_path)
                logger.info(f"Loaded FAISS index from disk: {faiss_path}")
            else:
                logger.warning(f"No FAISS index found at {faiss_path}")
                return []
            if os.path.exists(metadata_path):
                with open(metadata_path, 'rb') as f:
                    local_metadata = pickle.load(f)
                logger.info(f"Loaded metadata from disk: {metadata_path}")
            else:
                logger.warning(f"No metadata found at {metadata_path}")
                return []
            if os.path.exists(bm25_path):
                with open(bm25_path, 'rb') as f:
                    local_bm25_corpus = pickle.load(f)
                logger.info(f"Loaded BM25 corpus from disk: {bm25_path}")
                bm25 = BM25Okapi(local_bm25_corpus)
            else:
                logger.warning(f"No BM25 corpus found at {bm25_path}")
                return []

    if not local_metadata:
        logger.warning("No metadata available for search")
        return []

    query_embedding = model.encode(query, normalize_embeddings=True)
    query_embedding = np.array([query_embedding]).astype('float32')
    distances, indices = local_faiss_index.search(query_embedding, top_k)
    faiss_scores = 1 - distances[0]

    tokenized_query = query.lower().split()
    bm25_scores = bm25.get_scores(tokenized_query) if 'bm25' in locals() else np.zeros(len(local_metadata))
    bm25_top_indices = np.argsort(bm25_scores)[::-1][:top_k]
    bm25_top_scores = bm25_scores[bm25_top_indices]

    combined_results = {}
    for idx, score in zip(indices[0], faiss_scores):
        combined_results[idx] = combined_results.get(idx, 0) + 0.85 * score
    for idx, score in zip(bm25_top_indices, bm25_top_scores):
        combined_results[idx] = combined_results.get(idx, 0) + 0.15 * (score / max(bm25_scores) if max(bm25_scores) > 0 else 0)

    sorted_results = sorted(combined_results.items(), key=lambda x: x[1], reverse=True)[:top_k]
    results = []
    for idx, score in sorted_results:
        section = local_metadata[idx]
        results.append({
            'text': f"{section['title']}\n\n{section['content']}",
            'page': section['page'],
            'section': section['section'],
            'score': score
        })
    logger.info(f"Search returned {len(results)} results")
    return results

client = openai.OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.environ.get("GROQ_API_KEY")
)

def generate_llm_answer(query: str, context_chunks: List[Dict[str, Any]], session_id: str, company_folder: str = None) -> str:
    """Generate an answer using the LLM with context and chat history"""
    logger.info(f"Generating LLM answer for query: {query}, session: {session_id}")
    context = ""
    for i, chunk in enumerate(context_chunks[:5], 1):
        context += f"Chunk {i} (Page {chunk['page']}, Section {chunk['section']}):\n{chunk['text']}\n\n"

    history_context = ""
    if session_id in chat_history:
        recent_history = chat_history[session_id][-5:]
        for i, interaction in enumerate(recent_history, 1):
            history_context += f"Interaction {i}:\nUser Query: {interaction['query']}\nAssistant Answer: {interaction['answer']}\n\n"

    toc_path = os.path.join(INDEX_DIR, company_folder, f"{company_folder}_toc.txt") if company_folder else None
    toc_text = ""
    if toc_path and os.path.exists(toc_path):
        with open(toc_path, 'r', encoding='utf-8') as f:
            toc_text = f.read()
        logger.info(f"Loaded TOC from {toc_path}")

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

    for attempt in range(3):
        try:
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
            logger.info(f"Generated answer: {answer[:100]}...")
            if session_id not in chat_history:
                chat_history[session_id] = []
            chat_history[session_id].append({"query": query, "answer": answer})
            return answer
        except HTTPError as e:
            if e.response.status_code == 429:
                logger.warning(f"Groq API rate limit hit, retrying in {2**attempt} seconds")
                sleep(2**attempt)
            else:
                logger.error(f"Groq API error: {str(e)}")
                raise
    logger.error("Failed to generate answer after retries")
    raise HTTPException(status_code=429, detail="Groq API rate limit exceeded")

# --- Startup Event ---
@app.on_event("startup")
def startup_event():
    global model, redis_client
    logger.info("Starting up FastAPI application")
    model = SentenceTransformer(MODEL_NAME, device='cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Loaded SentenceTransformer model: {MODEL_NAME}")
    
    try:
        redis_client = redis.Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            db=REDIS_DB,
            decode_responses=False
        )
        redis_client.ping()
        logger.info("Connected to Redis")
    except Exception as e:
        logger.error(f"Failed to connect to Redis: {str(e)}", exc_info=True)
        redis_client = None

# --- Endpoints ---
@app.post("/upload_pdf", response_model=dict)
async def upload_pdf(
    file: UploadFile = File(...),
    company_name: str = Form(...),
    company_location: str = Form(...)
):
    """Upload and process a PDF file with company information"""
    logger.info(f"Uploading PDF: {file.filename}, Company: {company_name}, Location: {company_location}")
    company_folder = get_company_folder(company_name, company_location)
    company_index_dir = os.path.join(INDEX_DIR, company_folder)
    with get_company_lock(company_folder):
        logger.info(f"Acquired lock for company folder: {company_folder}")
        try:
            if not file.filename.endswith(('.pdf', '.xml')):
                logger.error("Invalid file type uploaded")
                raise HTTPException(status_code=400, detail="Only PDF or XML files are supported")
            if not company_name.strip() or not company_location.strip():
                logger.error("Missing company name or location")
                raise HTTPException(status_code=400, detail="Company name and location are required")
            process_pdf(file, file.filename, company_index_dir)
            logger.info(f"Successfully processed PDF: {file.filename}")
            return {
                "message": f"Document {file.filename} indexed successfully for {company_name} ({company_location})",
                "company_folder": company_folder
            }
        except HTTPException as e:
            logger.error(f"HTTP error during upload: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during upload: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """Process a query and return a response"""
    logger.info(f"Processing query: {request.query}, Session: {request.session_id}")
    try:
        company_folder = get_company_folder(request.company_name, request.company_location)
        company_index_dir = os.path.join(INDEX_DIR, company_folder)

        if not os.path.exists(company_index_dir):
            logger.error(f"No documents found for {company_folder}")
            raise HTTPException(status_code=404, detail=f"No documents found for {request.company_name} ({request.company_location})")

        results = hybrid_search(request.query, top_k=2, company_folder=company_folder)

        if not results:
            logger.warning(f"No search results for query: {request.query}")
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

        answer = generate_llm_answer(request.query, results, request.session_id, company_folder)
        top_result = results[0]
        logger.info(f"Query response generated, top result: {top_result['section']}")
        return QueryResponse(
            answer=answer,
            page=top_result['page'],
            section=top_result['section'],
            score=top_result['score']
        )

    except HTTPException as e:
        logger.error(f"HTTP error during query: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error during query: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.delete("/clear_history/{session_id}", response_model=dict)
async def clear_history(session_id: str):
    """Clear chat history for a specific session"""
    logger.info(f"Clearing chat history for session: {session_id}")
    try:
        if session_id in chat_history:
            del chat_history[session_id]
            logger.info(f"Chat history cleared for session: {session_id}")
            return {"message": f"Chat history for session {session_id} cleared successfully"}
        logger.warning(f"No chat history found for session: {session_id}")
        return {"message": f"No chat history found for session {session_id}"}
    except Exception as e:
        logger.error(f"Error clearing chat history: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error clearing chat history: {str(e)}")

@app.get("/context/{query}", response_model=List[ContextChunk])
async def get_context(
    query: str,
    company_name: str,
    company_location: str,
    top_k: int = 1
):
    """Get context chunks for a query"""
    logger.info(f"Fetching context for query: {query}, Company: {company_name}, Location: {company_location}")
    try:
        company_folder = get_company_folder(company_name, company_location)
        results = hybrid_search(query, top_k, company_folder)
        logger.info(f"Retrieved {len(results)} context chunks")
        return [
            ContextChunk(
                text=result['text'],
                page=result['page'],
                section=result['section'],
                score=result['score']
            ) for result in results
        ]
    except Exception as e:
        logger.error(f"Error retrieving context: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error retrieving context: {str(e)}")

from contextlib import asynccontextmanager

# --- Lifespan Event ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, redis_client
    logger.info("Starting up FastAPI application")
    model = SentenceTransformer(MODEL_NAME, device='cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Loaded SentenceTransformer model: {MODEL_NAME}")
    
    try:
        redis_client = redis.Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            db=REDIS_DB,
            decode_responses=False
        )
        redis_client.ping()
        logger.info("Connected to Redis")
    except Exception as e:
        logger.error(f"Failed to connect to Redis: {str(e)}", exc_info=True)
        redis_client = None
    
    yield  # Application runs here

app.lifespan = lifespan

if __name__ == "__main__":
    import uvicorn
    import multiprocessing
    logger.info("Starting Uvicorn server")
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8002,
        workers=multiprocessing.cpu_count()
    )