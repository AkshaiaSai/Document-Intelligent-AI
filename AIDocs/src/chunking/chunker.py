"""Smart text chunking with word-based overlap."""
import re
from typing import List, Dict
from src.utils import setup_logger, get_settings, get_paths, save_json, load_json, ensure_dir
import os
from pathlib import Path

logger = setup_logger(__name__)


class TextChunker:
    """Smart chunking with word-based size and overlap."""
    
    def __init__(self):
        """Initialize text chunker with settings from config."""
        settings = get_settings()
        chunking_config = settings.get('chunking', {})
        
        self.chunk_size_words = chunking_config.get('chunk_size_words', 600)
        self.overlap_words = chunking_config.get('overlap_words', 75)
        self.min_chunk_size = chunking_config.get('min_chunk_size_words', 100)
        
        self.paths = get_paths()
        ensure_dir(self.paths['processed_chunks_dir'])
        
        logger.info(f"Text Chunker initialized: {self.chunk_size_words} words, {self.overlap_words} overlap")
    
    def count_words(self, text: str) -> int:
        """Count words in text."""
        return len(text.split())
    
    def split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences using regex.
        
        Args:
            text: Input text
            
        Returns:
            List of sentences
        """
        # Simple sentence splitting (can be improved with NLTK)
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def create_chunks(self, text: str, metadata: Dict = None) -> List[Dict]:
        """
        Create word-based chunks with overlap.
        
        Args:
            text: Input text
            metadata: Optional metadata to attach to chunks
            
        Returns:
            List of chunk dictionaries
        """
        if not text.strip():
            return []
        
        # Split into sentences for better boundary detection
        sentences = self.split_into_sentences(text)
        
        chunks = []
        current_chunk = []
        current_word_count = 0
        chunk_index = 0
        
        for sentence in sentences:
            sentence_words = sentence.split()
            sentence_word_count = len(sentence_words)
            
            # If adding this sentence exceeds chunk size, save current chunk
            if current_word_count + sentence_word_count > self.chunk_size_words and current_chunk:
                chunk_text = ' '.join(current_chunk)
                
                chunks.append({
                    'chunk_id': chunk_index,
                    'text': chunk_text,
                    'word_count': current_word_count,
                    'metadata': metadata or {}
                })
                
                chunk_index += 1
                
                # Create overlap by keeping last N words
                overlap_text = ' '.join(current_chunk[-self.overlap_words:]) if len(current_chunk) > self.overlap_words else ' '.join(current_chunk)
                current_chunk = overlap_text.split()
                current_word_count = len(current_chunk)
            
            # Add sentence to current chunk
            current_chunk.extend(sentence_words)
            current_word_count += sentence_word_count
        
        # Add final chunk if it meets minimum size
        if current_chunk and current_word_count >= self.min_chunk_size:
            chunk_text = ' '.join(current_chunk)
            chunks.append({
                'chunk_id': chunk_index,
                'text': chunk_text,
                'word_count': current_word_count,
                'metadata': metadata or {}
            })
        
        logger.debug(f"Created {len(chunks)} chunks from {self.count_words(text)} words")
        return chunks
    
    def chunk_document_pages(self, pages: List[Dict], doc_metadata: Dict = None) -> List[Dict]:
        """
        Chunk document pages while preserving page numbers.
        
        Args:
            pages: List of page dictionaries with 'page_number' and 'text'
            doc_metadata: Document-level metadata
            
        Returns:
            List of chunks with page information
        """
        all_chunks = []
        
        for page in pages:
            page_number = page.get('page_number', 0)
            page_text = page.get('text', '')
            
            if not page_text.strip():
                continue
            
            # Create metadata for this page
            chunk_metadata = {
                'page_number': page_number,
                'extraction_method': page.get('method', 'unknown')
            }
            
            # Add document metadata if provided
            if doc_metadata:
                chunk_metadata.update({
                    'document_title': doc_metadata.get('title', ''),
                    'document_author': doc_metadata.get('author', ''),
                    'filename': doc_metadata.get('filename', '')
                })
            
            # Create chunks for this page
            page_chunks = self.create_chunks(page_text, chunk_metadata)
            all_chunks.extend(page_chunks)
        
        # Re-index chunks globally
        for i, chunk in enumerate(all_chunks):
            chunk['chunk_id'] = i
        
        logger.info(f"Created {len(all_chunks)} chunks from {len(pages)} pages")
        return all_chunks
    
    def process_extracted_document(self, extracted_json_path: str) -> List[Dict]:
        """
        Process an extracted document JSON file.
        
        Args:
            extracted_json_path: Path to extracted text JSON
            
        Returns:
            List of chunks
        """
        logger.info(f"Processing extracted document: {extracted_json_path}")
        
        # Load extracted document
        doc_data = load_json(extracted_json_path)
        
        metadata = doc_data.get('metadata', {})
        pages = doc_data.get('pages', [])
        
        # Create chunks
        chunks = self.chunk_document_pages(pages, metadata)
        
        # Save chunks
        output_filename = Path(extracted_json_path).stem.replace('_extracted', '_chunks.json')
        output_path = os.path.join(self.paths['processed_chunks_dir'], output_filename)
        
        save_json({
            'document_metadata': metadata,
            'chunks': chunks,
            'statistics': {
                'total_chunks': len(chunks),
                'total_pages': len(pages)
            }
        }, output_path)
        
        logger.info(f"Saved {len(chunks)} chunks to {output_path}")
        return chunks
    
    def process_directory(self, directory: str = None) -> List[Dict]:
        """
        Process all extracted documents in a directory.
        
        Args:
            directory: Directory path (defaults to extracted_texts_dir)
            
        Returns:
            List of all chunks from all documents
        """
        if directory is None:
            directory = self.paths['extracted_texts_dir']
        
        json_files = list(Path(directory).glob('*_extracted.json'))
        logger.info(f"Found {len(json_files)} extracted documents in {directory}")
        
        all_chunks = []
        for json_path in json_files:
            try:
                chunks = self.process_extracted_document(str(json_path))
                all_chunks.extend(chunks)
            except Exception as e:
                logger.error(f"Error processing {json_path}: {e}")
        
        logger.info(f"Processed {len(all_chunks)} total chunks from {len(json_files)} documents")
        return all_chunks
