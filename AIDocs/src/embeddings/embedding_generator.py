"""Embedding generation using Gemini API."""
from typing import List, Dict, Optional
import google.generativeai as genai
from tqdm import tqdm
import time
from src.utils import setup_logger, get_model_config, get_api_key, load_env

logger = setup_logger(__name__)


class EmbeddingGenerator:
    """Generate embeddings using Gemini API."""
    
    def __init__(self):
        """Initialize embedding generator."""
        # Load environment and API key
        load_env()
        api_key = get_api_key()
        genai.configure(api_key=api_key)
        
        # Load model configuration
        model_config = get_model_config()
        self.embedding_config = model_config.get('embedding_model', {})
        
        self.model_name = self.embedding_config.get('name', 'models/text-embedding-004')
        self.dimension = self.embedding_config.get('dimension', 768)
        self.batch_size = self.embedding_config.get('batch_size', 100)
        self.task_type = self.embedding_config.get('task_type', 'RETRIEVAL_DOCUMENT')
        
        logger.info(f"Embedding Generator initialized: {self.model_name}, dim={self.dimension}")
    
    def generate_embedding(self, text: str, task_type: str = None) -> List[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Input text
            task_type: Task type for embedding (RETRIEVAL_DOCUMENT or RETRIEVAL_QUERY)
            
        Returns:
            Embedding vector
        """
        if not text.strip():
            logger.warning("Empty text provided for embedding")
            return [0.0] * self.dimension
        
        try:
            task = task_type or self.task_type
            result = genai.embed_content(
                model=self.model_name,
                content=text,
                task_type=task
            )
            return result['embedding']
        
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return [0.0] * self.dimension
    
    def generate_embeddings_batch(
        self, 
        texts: List[str], 
        task_type: str = None,
        show_progress: bool = True
    ) -> List[List[float]]:
        """
        Generate embeddings for multiple texts with batching.
        
        Args:
            texts: List of input texts
            task_type: Task type for embeddings
            show_progress: Show progress bar
            
        Returns:
            List of embedding vectors
        """
        embeddings = []
        task = task_type or self.task_type
        
        # Process in batches
        iterator = range(0, len(texts), self.batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Generating embeddings")
        
        for i in iterator:
            batch = texts[i:i + self.batch_size]
            
            try:
                # Generate embeddings for batch
                batch_embeddings = []
                for text in batch:
                    emb = self.generate_embedding(text, task)
                    batch_embeddings.append(emb)
                    
                    # Small delay to avoid rate limiting
                    time.sleep(0.1)
                
                embeddings.extend(batch_embeddings)
                
            except Exception as e:
                logger.error(f"Error in batch {i//self.batch_size}: {e}")
                # Add zero vectors for failed batch
                embeddings.extend([[0.0] * self.dimension] * len(batch))
        
        logger.info(f"Generated {len(embeddings)} embeddings")
        return embeddings
    
    def embed_chunks(self, chunks: List[Dict], show_progress: bool = True) -> List[Dict]:
        """
        Generate embeddings for chunks and attach them.
        
        Args:
            chunks: List of chunk dictionaries with 'text' field
            show_progress: Show progress bar
            
        Returns:
            Chunks with embeddings attached
        """
        logger.info(f"Embedding {len(chunks)} chunks")
        
        # Extract texts
        texts = [chunk['text'] for chunk in chunks]
        
        # Generate embeddings
        embeddings = self.generate_embeddings_batch(
            texts, 
            task_type='RETRIEVAL_DOCUMENT',
            show_progress=show_progress
        )
        
        # Attach embeddings to chunks
        for chunk, embedding in zip(chunks, embeddings):
            chunk['embedding'] = embedding
        
        logger.info("Embeddings attached to chunks")
        return chunks
    
    def embed_query(self, query: str) -> List[float]:
        """
        Generate embedding for a query.
        
        Args:
            query: Query text
            
        Returns:
            Query embedding vector
        """
        return self.generate_embedding(query, task_type='RETRIEVAL_QUERY')
