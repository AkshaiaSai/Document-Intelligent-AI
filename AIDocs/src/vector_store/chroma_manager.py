"""ChromaDB vector store manager."""
from typing import List, Dict, Optional, Tuple
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
import uuid
from src.utils import setup_logger, get_paths, get_settings, ensure_dir
from src.embeddings import EmbeddingGenerator

logger = setup_logger(__name__)


class ChromaManager:
    """Manage ChromaDB vector store for document chunks."""
    
    def __init__(self, collection_name: str = "pdf_documents"):
        """
        Initialize ChromaDB manager.
        
        Args:
            collection_name: Name of the collection
        """
        self.paths = get_paths()
        self.settings_config = get_settings()
        self.collection_name = collection_name
        
        # Ensure persist directory exists
        persist_dir = self.paths['chroma_persist_dir']
        ensure_dir(persist_dir)
        
        # Initialize ChromaDB client with persistence
        self.client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "PDF document chunks with embeddings"}
        )
        
        # Initialize embedding generator
        self.embedding_generator = EmbeddingGenerator()
        
        logger.info(f"ChromaDB initialized: collection='{collection_name}', persist_dir='{persist_dir}'")
    
    def add_chunks(self, chunks: List[Dict], batch_size: int = 100) -> None:
        """
        Add chunks to the vector store.
        
        Args:
            chunks: List of chunk dictionaries with text, metadata, and optionally embeddings
            batch_size: Batch size for adding to ChromaDB
        """
        if not chunks:
            logger.warning("No chunks to add")
            return
        
        logger.info(f"Adding {len(chunks)} chunks to ChromaDB")
        
        # Check if chunks have embeddings
        if 'embedding' not in chunks[0]:
            logger.info("Chunks don't have embeddings, generating them...")
            chunks = self.embedding_generator.embed_chunks(chunks)
        
        # Prepare data for ChromaDB
        ids = []
        documents = []
        embeddings = []
        metadatas = []
        
        for chunk in chunks:
            # Generate unique ID
            chunk_id = str(uuid.uuid4())
            ids.append(chunk_id)
            
            # Extract text
            documents.append(chunk['text'])
            
            # Extract embedding
            embeddings.append(chunk['embedding'])
            
            # Prepare metadata (ChromaDB requires simple types)
            metadata = chunk.get('metadata', {})
            clean_metadata = {
                'chunk_id': chunk.get('chunk_id', 0),
                'page_number': metadata.get('page_number', 0),
                'document_title': metadata.get('document_title', ''),
                'filename': metadata.get('filename', ''),
                'word_count': chunk.get('word_count', 0)
            }
            metadatas.append(clean_metadata)
        
        # Add to ChromaDB in batches
        for i in range(0, len(chunks), batch_size):
            batch_end = min(i + batch_size, len(chunks))
            
            self.collection.add(
                ids=ids[i:batch_end],
                documents=documents[i:batch_end],
                embeddings=embeddings[i:batch_end],
                metadatas=metadatas[i:batch_end]
            )
            
            logger.debug(f"Added batch {i//batch_size + 1}: {batch_end - i} chunks")
        
        logger.info(f"Successfully added {len(chunks)} chunks to ChromaDB")
    
    def search(
        self, 
        query: str, 
        n_results: int = 8,
        where: Optional[Dict] = None,
        where_document: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Search for relevant chunks using semantic similarity.
        
        Args:
            query: Search query
            n_results: Number of results to return
            where: Metadata filter
            where_document: Document content filter
            
        Returns:
            List of result dictionaries with text, metadata, and distance
        """
        # Generate query embedding
        query_embedding = self.embedding_generator.embed_query(query)
        
        # Search in ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where,
            where_document=where_document
        )
        
        # Format results
        formatted_results = []
        if results['ids'] and results['ids'][0]:
            for i in range(len(results['ids'][0])):
                formatted_results.append({
                    'id': results['ids'][0][i],
                    'text': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'distance': results['distances'][0][i] if 'distances' in results else None,
                    'similarity': 1 - results['distances'][0][i] if 'distances' in results else None
                })
        
        logger.info(f"Found {len(formatted_results)} results for query")
        return formatted_results
    
    def hybrid_search(
        self,
        query: str,
        n_results: int = 8,
        semantic_weight: float = 0.7,
        keyword_weight: float = 0.3
    ) -> List[Dict]:
        """
        Perform hybrid search combining semantic and keyword matching.
        
        Args:
            query: Search query
            n_results: Number of results to return
            semantic_weight: Weight for semantic similarity
            keyword_weight: Weight for keyword matching
            
        Returns:
            List of ranked results
        """
        # Get more results for reranking
        search_n = n_results * 2
        
        # Semantic search
        semantic_results = self.search(query, n_results=search_n)
        
        # Keyword search using where_document
        keyword_results = self.collection.query(
            query_texts=[query],
            n_results=search_n
        )
        
        # Combine and rerank results
        result_scores = {}
        
        # Score semantic results
        for i, result in enumerate(semantic_results):
            doc_id = result['id']
            # Normalize rank to 0-1 score (higher rank = lower score)
            semantic_score = 1 - (i / len(semantic_results))
            result_scores[doc_id] = {
                'result': result,
                'score': semantic_score * semantic_weight
            }
        
        # Add keyword scores
        if keyword_results['ids'] and keyword_results['ids'][0]:
            for i, doc_id in enumerate(keyword_results['ids'][0]):
                keyword_score = 1 - (i / len(keyword_results['ids'][0]))
                
                if doc_id in result_scores:
                    result_scores[doc_id]['score'] += keyword_score * keyword_weight
                else:
                    result_scores[doc_id] = {
                        'result': {
                            'id': doc_id,
                            'text': keyword_results['documents'][0][i],
                            'metadata': keyword_results['metadatas'][0][i],
                            'distance': keyword_results['distances'][0][i] if 'distances' in keyword_results else None
                        },
                        'score': keyword_score * keyword_weight
                    }
        
        # Sort by combined score
        ranked_results = sorted(
            result_scores.values(),
            key=lambda x: x['score'],
            reverse=True
        )[:n_results]
        
        # Extract results
        final_results = [item['result'] for item in ranked_results]
        
        logger.info(f"Hybrid search returned {len(final_results)} results")
        return final_results
    
    def get_collection_stats(self) -> Dict:
        """Get statistics about the collection."""
        count = self.collection.count()
        return {
            'collection_name': self.collection_name,
            'total_chunks': count,
            'persist_directory': self.paths['chroma_persist_dir']
        }
    
    def clear_collection(self) -> None:
        """Clear all data from the collection."""
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.create_collection(
            name=self.collection_name,
            metadata={"description": "PDF document chunks with embeddings"}
        )
        logger.info(f"Cleared collection '{self.collection_name}'")
    
    def delete_by_filename(self, filename: str) -> None:
        """
        Delete all chunks from a specific file.
        
        Args:
            filename: Name of the file to delete
        """
        self.collection.delete(
            where={"filename": filename}
        )
        logger.info(f"Deleted chunks from file: {filename}")
