# HR Policy Document Chatbot

A FastAPI-based chatbot that can answer questions about HR policy documents using PDF processing, semantic search, and LLM integration.

## Features

- PDF document processing and indexing
- Hybrid search using FAISS and BM25
- LLM-powered question answering
- Chat history management
- Context retrieval

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file with your API keys:
```
GROQ_API_KEY=your_groq_api_key_here
```

## Project Structure

```
app/
├── __init__.py
├── main.py              # FastAPI application
├── config.py            # Configuration settings
├── models/
│   ├── __init__.py
│   └── schemas.py       # Pydantic models
├── services/
│   ├── __init__.py
│   ├── pdf_service.py   # PDF processing
│   ├── search_service.py # Search functionality
│   └── llm_service.py   # LLM integration
└── utils/
    ├── __init__.py
    └── pdf_parser.py    # PDF parsing utilities
```

## Usage

1. Start the server:
```bash
python -m app.main
```

2. API Endpoints:
- `POST /upload_pdf`: Upload and process a PDF file
- `POST /query`: Ask questions about the uploaded documents
- `DELETE /clear_history/{session_id}`: Clear chat history for a session
- `GET /context/{query}`: Get context chunks for a query

## API Documentation

Once the server is running, visit:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Dependencies

- FastAPI: Web framework
- Uvicorn: ASGI server
- PDFPlumber: PDF text extraction
- FAISS: Vector similarity search
- Sentence Transformers: Text embeddings
- BM25: Keyword-based search
- OpenAI/Groq: LLM integration
- PyTorch: Deep learning framework 