"""Hybrid retrieval combining semantic and keyword search."""
from typing import List, Dict, Set
from src.utils import setup_logger, get_settings
from src.vector_store import ChromaManager
from src.qa_pipeline.query_expander import QueryExpander

logger = setup_logger(__name__)


class HybridRetriever:
    """Retrieve relevant chunks using hybrid search with query expansion."""
    
    def __init__(self, chroma_manager: ChromaManager = None):
        """
        Initialize hybrid retriever.
        
        Args:
            chroma_manager: ChromaDB manager instance (creates new if None)
        """
        self.chroma_manager = chroma_manager or ChromaManager()
        self.query_expander = QueryExpander()
        
        # Load settings
        settings = get_settings()
        self.retrieval_config = settings.get('retrieval', {})
        
        self.top_k = self.retrieval_config.get('top_k', 8)
        self.similarity_threshold = self.retrieval_config.get('similarity_threshold', 0.3)
        self.use_hybrid = self.retrieval_config.get('use_hybrid_search', True)
        self.semantic_weight = self.retrieval_config.get('semantic_weight', 0.7)
        self.keyword_weight = self.retrieval_config.get('keyword_weight', 0.3)
        
        logger.info(f"Hybrid Retriever initialized: top_k={self.top_k}, threshold={self.similarity_threshold}")
    
    def deduplicate_results(self, results: List[Dict]) -> List[Dict]:
        """
        Remove duplicate results based on text similarity.
        
        Args:
            results: List of result dictionaries
            
        Returns:
            Deduplicated results
        """
        seen_texts: Set[str] = set()
        unique_results = []
        
        for result in results:
            text = result['text']
            # Use first 100 characters as fingerprint
            fingerprint = text[:100]
            
            if fingerprint not in seen_texts:
                seen_texts.add(fingerprint)
                unique_results.append(result)
        
        if len(unique_results) < len(results):
            logger.debug(f"Removed {len(results) - len(unique_results)} duplicate results")
        
        return unique_results
    
    def filter_by_threshold(self, results: List[Dict]) -> List[Dict]:
        """
        Filter results by similarity threshold.
        
        Args:
            results: List of result dictionaries with 'similarity' field
            
        Returns:
            Filtered results
        """
        filtered = [r for r in results if r.get('similarity', 0) >= self.similarity_threshold]
        
        if len(filtered) < len(results):
            logger.debug(f"Filtered out {len(results) - len(filtered)} results below threshold")
        
        return filtered
    
    def retrieve_with_expansion(self, query: str, top_k: int = None) -> List[Dict]:
        """
        Retrieve chunks using query expansion and hybrid search.
        
        Args:
            query: User query
            top_k: Number of results to return (default from config)
            
        Returns:
            List of relevant chunks
        """
        k = top_k or self.top_k
        
        # Expand query
        query_variations = self.query_expander.expand_query(query)
        logger.info(f"Using {len(query_variations)} query variations")
        
        # Retrieve for each variation
        all_results = []
        for variation in query_variations:
            if self.use_hybrid:
                results = self.chroma_manager.hybrid_search(
                    variation,
                    n_results=k,
                    semantic_weight=self.semantic_weight,
                    keyword_weight=self.keyword_weight
                )
            else:
                results = self.chroma_manager.search(variation, n_results=k)
            
            all_results.extend(results)
        
        # Deduplicate
        unique_results = self.deduplicate_results(all_results)
        
        # Filter by threshold
        filtered_results = self.filter_by_threshold(unique_results)
        
        # Sort by similarity and take top k
        sorted_results = sorted(
            filtered_results,
            key=lambda x: x.get('similarity', 0),
            reverse=True
        )[:k]
        
        logger.info(f"Retrieved {len(sorted_results)} unique chunks")
        return sorted_results
    
    def retrieve(self, query: str, top_k: int = None) -> List[Dict]:
        """
        Main retrieval method.
        
        Args:
            query: User query
            top_k: Number of results to return
            
        Returns:
            List of relevant chunks
        """
        return self.retrieve_with_expansion(query, top_k)
    
    def format_context(self, results: List[Dict]) -> str:
        """
        Format retrieved chunks into context string.
        
        Args:
            results: List of retrieved chunks
            
        Returns:
            Formatted context string
        """
        if not results:
            return "No relevant context found."
        
        context_parts = []
        for i, result in enumerate(results):
            metadata = result.get('metadata', {})
            page_num = metadata.get('page_number', 'Unknown')
            doc_title = metadata.get('document_title', 'Unknown Document')
            
            context_parts.append(
                f"[Source {i+1} - {doc_title}, Page {page_num}]\n{result['text']}"
            )
        
        return "\n\n".join(context_parts)
