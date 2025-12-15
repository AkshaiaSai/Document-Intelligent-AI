# Technical Documentation

## Architecture Overview

The Context-Aware RAG Agent is built with a modular architecture consisting of six main components:

### 1. PDF Extraction Module

**Location**: `src/pdf_extraction/`

**Components**:
- `PDFExtractor`: Main extraction class using pdfplumber
- `OCRHandler`: OCR processing for scanned documents

**Flow**:
1. Attempt text extraction with pdfplumber
2. Analyze extraction quality (character count per page)
3. If quality is low (<10% pages with text), trigger OCR
4. OCR converts PDF pages to images (300 DPI)
5. Pytesseract extracts text from images
6. Results saved as JSON with metadata

**Key Features**:
- Automatic OCR fallback
- Page number preservation
- Document metadata extraction
- Batch processing support

### 2. Chunking Module

**Location**: `src/chunking/`

**Algorithm**:
```python
1. Split text into sentences
2. Accumulate sentences until reaching 600 words
3. Create chunk with metadata
4. Keep last 75 words for overlap
5. Continue with next chunk
```

**Metadata Preserved**:
- Page number
- Document title
- Filename
- Extraction method
- Word count

**Configuration** (`config/settings.yaml`):
```yaml
chunking:
  chunk_size_words: 600
  overlap_words: 75
  min_chunk_size_words: 100
```

### 3. Embedding Module

**Location**: `src/embeddings/`

**Model**: Google Gemini `text-embedding-004`
- Dimension: 768
- Task types: RETRIEVAL_DOCUMENT, RETRIEVAL_QUERY

**Features**:
- Batch processing (100 chunks per batch)
- Rate limiting (0.1s delay between requests)
- Error handling with zero-vector fallback
- Progress tracking

**API Usage**:
```python
from src.embeddings import EmbeddingGenerator

generator = EmbeddingGenerator()

# Single embedding
embedding = generator.generate_embedding("text", task_type="RETRIEVAL_DOCUMENT")

# Batch embeddings
embeddings = generator.generate_embeddings_batch(texts)

# Embed chunks (adds embedding field)
chunks_with_embeddings = generator.embed_chunks(chunks)
```

### 4. Vector Store Module

**Location**: `src/vector_store/`

**Technology**: ChromaDB with persistent storage

**Features**:
- Persistent storage in `data/embeddings/chroma_db`
- Metadata filtering
- Hybrid search (semantic + keyword)
- Batch operations

**Search Methods**:

1. **Semantic Search**:
   - Uses query embeddings
   - Cosine similarity
   - Returns top-k results

2. **Hybrid Search**:
   - Combines semantic (70%) and keyword (30%)
   - Reranks results by combined score
   - Deduplicates results

**API Usage**:
```python
from src.vector_store import ChromaManager

manager = ChromaManager()

# Add chunks
manager.add_chunks(chunks)

# Semantic search
results = manager.search("query", n_results=8)

# Hybrid search
results = manager.hybrid_search("query", n_results=8)

# Filter by metadata
results = manager.search("query", where={"page_number": 5})

# Get stats
stats = manager.get_collection_stats()
```

### 5. QA Pipeline Module

**Location**: `src/qa_pipeline/`

**Components**:

#### Query Expander
- Generates 3-7 query variations using Gemini
- Preserves original meaning
- Uses synonyms and reformulations
- Temperature: 0.7 for diversity

#### Hybrid Retriever
- Expands query into variations
- Retrieves chunks for each variation
- Deduplicates results
- Filters by similarity threshold (0.3)
- Returns top-8 chunks

#### Answer Generator
- Uses Gemini 1.5 Flash
- Temperature: 0.3 for consistency
- Context-aware prompting
- Extracts citations from metadata
- Grounds answers in retrieved context

**Prompt Template**:
```
You are a helpful AI assistant that answers questions based on provided document context.

INSTRUCTIONS:
1. Answer using ONLY the provided context
2. If answer not in context, say so
3. Include page citations [Source X, Page Y]
4. Be detailed and comprehensive
5. Quote relevant parts when appropriate
6. Do not use external knowledge

CONTEXT:
{context}

QUESTION: {question}

ANSWER:
```

### 6. Main Pipeline

**Location**: `src/main.py`

**Class**: `RAGPipeline`

**Methods**:

```python
# Process single PDF
stats = pipeline.process_pdf("path/to/pdf")

# Process multiple PDFs
results = pipeline.process_multiple_pdfs(["pdf1", "pdf2"])

# Ask question
answer = pipeline.ask_question("What is X?", top_k=8)

# Get statistics
stats = pipeline.get_stats()

# Clear database
pipeline.clear_database()

# Cleanup temp files
pipeline.cleanup()
```

## Configuration Files

### paths_config.json

Defines all data directories:
```json
{
  "data_dir": "data",
  "raw_pdfs_dir": "data/raw_pdfs",
  "extracted_texts_dir": "data/extracted_texts",
  "processed_chunks_dir": "data/processed_chunks",
  "embeddings_dir": "data/embeddings",
  "temp_images_dir": "data/temp_images",
  "chroma_persist_dir": "data/embeddings/chroma_db"
}
```

### model_config.json

Model configurations:
```json
{
  "embedding_model": {
    "name": "models/text-embedding-004",
    "dimension": 768,
    "batch_size": 100,
    "task_type": "RETRIEVAL_DOCUMENT"
  },
  "llm_model": {
    "name": "gemini-1.5-flash",
    "temperature": 0.3,
    "max_output_tokens": 2048,
    "top_p": 0.95,
    "top_k": 40
  },
  "query_expansion_model": {
    "name": "gemini-1.5-flash",
    "temperature": 0.7,
    "max_output_tokens": 512
  }
}
```

### settings.yaml

Pipeline parameters:
```yaml
chunking:
  chunk_size_words: 600
  overlap_words: 75
  min_chunk_size_words: 100

retrieval:
  top_k: 8
  similarity_threshold: 0.3
  use_hybrid_search: true
  keyword_weight: 0.3
  semantic_weight: 0.7

query_expansion:
  enabled: true
  min_variations: 3
  max_variations: 7

ocr:
  enabled: true
  language: "eng"
  dpi: 300
  preprocessing: true

logging:
  level: "INFO"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  file: "logs/rag_agent.log"
```

## Data Flow

### Document Processing

```
PDF File
  ↓
PDFExtractor.process_pdf()
  ↓
{metadata, pages: [{page_number, text, method}]}
  ↓
TextChunker.chunk_document_pages()
  ↓
[{chunk_id, text, word_count, metadata}]
  ↓
EmbeddingGenerator.embed_chunks()
  ↓
[{chunk_id, text, metadata, embedding}]
  ↓
ChromaManager.add_chunks()
  ↓
ChromaDB Collection
```

### Question Answering

```
User Question
  ↓
QueryExpander.expand_query()
  ↓
[query_variation_1, query_variation_2, ...]
  ↓
HybridRetriever.retrieve_with_expansion()
  ↓
For each variation:
  - ChromaManager.hybrid_search()
  - Combine results
  - Deduplicate
  - Filter by threshold
  ↓
Top-K Chunks
  ↓
AnswerGenerator.answer_question()
  ↓
{answer, citations, num_sources}
```

## Performance Optimization

### Embedding Generation
- Batch size: 100 chunks
- Rate limiting: 0.1s between requests
- Parallel processing: Not implemented (API limitation)

### Vector Search
- ChromaDB uses HNSW index
- Fast approximate nearest neighbor search
- Metadata filtering at query time

### Chunking
- Sentence-based splitting for better boundaries
- Overlap prevents context loss
- Configurable chunk size

## Error Handling

### PDF Extraction
- Falls back to OCR if text extraction fails
- Handles corrupted PDFs gracefully
- Logs extraction method used

### Embedding Generation
- Returns zero vectors on API failure
- Continues processing remaining chunks
- Logs errors for debugging

### Vector Search
- Returns empty results if no matches
- Handles missing metadata gracefully
- Validates similarity scores

## Logging

All modules use centralized logging:

```python
from src.utils import setup_logger

logger = setup_logger(__name__)
logger.info("Information message")
logger.warning("Warning message")
logger.error("Error message")
```

Logs are written to:
- Console: INFO level
- File: DEBUG level (`logs/rag_agent.log`)

## Testing

### Manual Testing

1. **PDF Extraction**:
```bash
python -c "from src.pdf_extraction import PDFExtractor; e = PDFExtractor(); print(e.process_pdf('test.pdf'))"
```

2. **Chunking**:
```bash
python -c "from src.chunking import TextChunker; c = TextChunker(); print(c.process_directory())"
```

3. **End-to-End**:
```bash
python src/main.py test.pdf "What is the main topic?"
```

### Unit Testing

Create `tests/` directory with pytest:

```python
# tests/test_chunker.py
from src.chunking import TextChunker

def test_chunking():
    chunker = TextChunker()
    text = "Sample text " * 1000
    chunks = chunker.create_chunks(text)
    assert len(chunks) > 0
    assert all(c['word_count'] <= 650 for c in chunks)
```

## Deployment

### Local Deployment

```bash
streamlit run src/interface/app.py --server.port 8501
```

### Production Deployment

1. **Docker**:
```dockerfile
FROM python:3.9
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
RUN apt-get update && apt-get install -y tesseract-ocr
COPY . .
CMD ["streamlit", "run", "src/interface/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

2. **Environment Variables**:
```bash
export GOOGLE_API_KEY=your_key
export STREAMLIT_SERVER_PORT=8501
```

## Troubleshooting

### Common Issues

1. **Import Errors**:
   - Ensure you're in the project root
   - Check Python path includes `src/`

2. **API Rate Limits**:
   - Increase delay in `embedding_generator.py`
   - Reduce batch size

3. **Memory Issues**:
   - Process PDFs one at a time
   - Reduce chunk size
   - Clear database regularly

4. **OCR Accuracy**:
   - Increase DPI (300 → 600)
   - Enable preprocessing
   - Use language-specific models

## API Reference

See inline documentation in each module for detailed API reference.

---

For questions or issues, check the logs in `logs/rag_agent.log`.
