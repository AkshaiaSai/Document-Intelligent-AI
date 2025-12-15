"""Main orchestration for the RAG pipeline."""
from pathlib import Path
from typing import List, Dict, Optional
from src.utils import setup_logger, get_paths
from src.pdf_extraction import PDFExtractor
from src.chunking import TextChunker
from src.embeddings import EmbeddingGenerator
from src.vector_store import ChromaManager
from src.qa_pipeline import HybridRetriever, AnswerGenerator

logger = setup_logger(__name__)


class RAGPipeline:
    """Main RAG pipeline orchestrator."""
    
    def __init__(self):
        """Initialize RAG pipeline components."""
        logger.info("Initializing RAG Pipeline...")
        
        self.paths = get_paths()
        self.pdf_extractor = PDFExtractor()
        self.chunker = TextChunker()
        self.embedding_generator = EmbeddingGenerator()
        self.chroma_manager = ChromaManager()
        self.retriever = HybridRetriever(self.chroma_manager)
        self.answer_generator = AnswerGenerator()
        
        logger.info("RAG Pipeline initialized successfully")
    
    def process_pdf(self, pdf_path: str) -> Dict:
        """
        Process a single PDF through the entire pipeline.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Processing statistics
        """
        logger.info(f"Processing PDF: {pdf_path}")
        
        # Extract text
        extraction_result = self.pdf_extractor.process_pdf(pdf_path)
        
        # Chunk text
        chunks = self.chunker.chunk_document_pages(
            extraction_result['pages'],
            extraction_result['metadata']
        )
        
        # Generate embeddings and add to vector store
        self.chroma_manager.add_chunks(chunks)
        
        stats = {
            'filename': Path(pdf_path).name,
            'pages': extraction_result['statistics']['total_pages'],
            'chunks': len(chunks),
            'characters': extraction_result['statistics']['total_characters']
        }
        
        logger.info(f"PDF processed: {stats}")
        return stats
    
    def process_multiple_pdfs(self, pdf_paths: List[str]) -> List[Dict]:
        """
        Process multiple PDFs.
        
        Args:
            pdf_paths: List of PDF file paths
            
        Returns:
            List of processing statistics
        """
        results = []
        for pdf_path in pdf_paths:
            try:
                stats = self.process_pdf(pdf_path)
                results.append(stats)
            except Exception as e:
                logger.error(f"Error processing {pdf_path}: {e}")
                results.append({
                    'filename': Path(pdf_path).name,
                    'error': str(e)
                })
        
        return results
    
    def ask_question(self, question: str, top_k: int = 8) -> Dict:
        """
        Answer a question using the RAG pipeline.
        
        Args:
            question: User question
            top_k: Number of chunks to retrieve
            
        Returns:
            Answer dictionary with citations
        """
        logger.info(f"Answering question: {question}")
        
        # Retrieve relevant chunks
        retrieved_chunks = self.retriever.retrieve(question, top_k=top_k)
        
        if not retrieved_chunks:
            return {
                'question': question,
                'answer': "I couldn't find any relevant information in the documents to answer this question.",
                'citations': [],
                'num_sources': 0
            }
        
        # Generate answer
        answer_result = self.answer_generator.answer_question(question, retrieved_chunks)
        
        return answer_result
    
    def get_stats(self) -> Dict:
        """Get pipeline statistics."""
        chroma_stats = self.chroma_manager.get_collection_stats()
        return {
            'total_chunks': chroma_stats['total_chunks'],
            'collection_name': chroma_stats['collection_name']
        }
    
    def clear_database(self) -> None:
        """Clear all data from the vector database."""
        self.chroma_manager.clear_collection()
        logger.info("Database cleared")
    
    def cleanup(self) -> None:
        """Clean up temporary files."""
        self.pdf_extractor.cleanup()
        logger.info("Cleanup completed")


def main():
    """Main entry point for CLI usage."""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python main.py <pdf_path> [question]")
        sys.exit(1)
    
    # Initialize pipeline
    pipeline = RAGPipeline()
    
    # Process PDF
    pdf_path = sys.argv[1]
    print(f"Processing {pdf_path}...")
    stats = pipeline.process_pdf(pdf_path)
    print(f"Processed: {stats}")
    
    # Answer question if provided
    if len(sys.argv) > 2:
        question = " ".join(sys.argv[2:])
        print(f"\nQuestion: {question}")
        result = pipeline.ask_question(question)
        print(f"\nAnswer: {result['answer']}")
        print(f"\nSources ({result['num_sources']}):")
        for citation in result['citations']:
            print(f"  - {citation['document_title']}, Page {citation['page_number']}")
    
    # Cleanup
    pipeline.cleanup()


if __name__ == "__main__":
    main()
