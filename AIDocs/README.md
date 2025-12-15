# Context-Aware RAG Agent 📚

Advanced PDF Question Answering system powered by Gemini AI with OCR support, smart chunking, and hybrid retrieval.

## ✨ Features

- **📄 PDF Processing**: Extract text from PDFs with automatic OCR fallback for scanned documents
- **🧩 Smart Chunking**: 600-word chunks with 75-word overlap for optimal context preservation
- **🔍 Hybrid Search**: Combines semantic similarity and keyword matching
- **🎯 Query Expansion**: Generates 3-7 query variations for improved retrieval
- **🤖 Gemini AI**: Powered by Google's Gemini 1.5 Flash for embeddings and answer generation
- **📊 ChromaDB**: Persistent vector storage with metadata filtering
- **💬 Modern UI**: Beautiful Streamlit interface with real-time processing
- **📖 Citations**: Grounded answers with page-level source citations

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Google Gemini API key
- Tesseract OCR (for scanned PDFs)

### Installation

1. **Clone or navigate to the project directory**

```bash
cd /Users/akshaiasai/Desktop/AIDocs
```

2. **Install dependencies**

```bash
pip install -r requirements.txt
```

3. **Install Tesseract OCR** (for OCR support)

```bash
# macOS
brew install tesseract

# Ubuntu/Debian
sudo apt-get install tesseract-ocr

# Windows
# Download from: https://github.com/UB-Mannheim/tesseract/wiki
```

4. **Set up environment variables**

Create a `.env` file in the project root:

```env
GOOGLE_API_KEY=your_gemini_api_key_here
```

### Running the Application

**Streamlit Web Interface** (Recommended):

```bash
streamlit run src/interface/app.py
```

**Command Line Interface**:

```bash
# Process a PDF
python src/main.py path/to/document.pdf

# Process and ask a question
python src/main.py path/to/document.pdf "What is the main topic?"
```

## 📁 Project Structure

```
context_aware_rag_agent/
├── config/                      # Configuration files
│   ├── paths_config.json        # Directory paths
│   ├── model_config.json        # Model settings
│   └── settings.yaml            # Pipeline parameters
├── data/                        # Data storage (auto-created)
│   ├── raw_pdfs/               # Uploaded PDFs
│   ├── extracted_texts/        # Extracted text
│   ├── processed_chunks/       # Chunked data
│   └── embeddings/             # ChromaDB storage
├── src/                        # Source code
│   ├── pdf_extraction/         # PDF & OCR processing
│   ├── chunking/               # Text chunking
│   ├── embeddings/             # Embedding generation
│   ├── vector_store/           # ChromaDB management
│   ├── qa_pipeline/            # Retrieval & generation
│   ├── utils/                  # Utilities
│   ├── interface/              # Streamlit UI
│   └── main.py                 # Main orchestrator
└── requirements.txt            # Dependencies
```

## 🎯 How It Works

```
User Uploads PDF
        ↓
Text Extraction (pdfplumber + OCR)
        ↓
Smart Chunking (600 words, 75 overlap)
        ↓
Embedding Creation (768-dim Gemini vectors)
        ↓
Vector Store (ChromaDB)
        ↓
User Query
        ↓
Hybrid Retrieval (Semantic + Keyword)
   • Query expansion (3-7 variations)
   • Top-8 most relevant chunks
   • Dynamic similarity threshold (0.3)
        ↓
LLM Answer Generation (Gemini 1.5 Flash)
   • Context + Question → Single prompt
   • Grounded answers with page citations
        ↓
Display Answer + Sources
```

## ⚙️ Configuration

### Model Settings (`config/model_config.json`)

- **Embedding Model**: `text-embedding-004` (768 dimensions)
- **LLM Model**: `gemini-1.5-flash`
- **Temperature**: 0.3 (for consistent answers)

### Pipeline Settings (`config/settings.yaml`)

- **Chunk Size**: 600 words
- **Overlap**: 75 words
- **Top-K Retrieval**: 8 chunks
- **Similarity Threshold**: 0.3
- **Query Variations**: 3-7

## 🎨 Using the Web Interface

1. **Upload PDFs**: Drag and drop or select PDF files in the sidebar
2. **Process**: Click "Process PDFs" to extract, chunk, and embed
3. **Ask Questions**: Type questions in the chat interface
4. **View Answers**: Get grounded answers with source citations
5. **Check Stats**: Monitor processed documents and chunks

## 📚 API Usage

```python
from src.main import RAGPipeline

# Initialize pipeline
pipeline = RAGPipeline()

# Process a PDF
stats = pipeline.process_pdf("document.pdf")
print(f"Processed {stats['chunks']} chunks from {stats['pages']} pages")

# Ask a question
result = pipeline.ask_question("What is the main conclusion?")
print(f"Answer: {result['answer']}")
print(f"Sources: {result['num_sources']}")

# View citations
for citation in result['citations']:
    print(f"- {citation['document_title']}, Page {citation['page_number']}")

# Cleanup
pipeline.cleanup()
```

## 🔧 Advanced Usage

### Custom Chunking

Edit `config/settings.yaml`:

```yaml
chunking:
  chunk_size_words: 800  # Increase chunk size
  overlap_words: 100     # Increase overlap
```

### Adjust Retrieval

```yaml
retrieval:
  top_k: 10              # Retrieve more chunks
  similarity_threshold: 0.4  # Higher threshold
  semantic_weight: 0.8   # More weight on semantic search
  keyword_weight: 0.2
```

### Disable Query Expansion

```yaml
query_expansion:
  enabled: false
```

## 🐛 Troubleshooting

**OCR not working?**
- Ensure Tesseract is installed: `tesseract --version`
- Check OCR settings in `config/settings.yaml`

**API errors?**
- Verify your Gemini API key in `.env`
- Check API quota and rate limits

**Slow processing?**
- Reduce batch size in `config/model_config.json`
- Disable OCR for text-based PDFs

**No results found?**
- Lower similarity threshold in `config/settings.yaml`
- Check if PDFs were processed successfully

## 📝 License

MIT License - feel free to use and modify!

## 🙏 Acknowledgments

- **Google Gemini**: For powerful embeddings and LLM
- **ChromaDB**: For efficient vector storage
- **pdfplumber**: For PDF text extraction
- **Streamlit**: For the beautiful UI framework

---

**Built with ❤️ using Gemini AI**
