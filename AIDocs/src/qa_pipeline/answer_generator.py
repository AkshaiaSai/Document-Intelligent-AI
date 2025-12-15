"""Answer generation using Gemini LLM."""
from typing import List, Dict, Tuple
import google.generativeai as genai
from src.utils import setup_logger, get_model_config, get_api_key, load_env

logger = setup_logger(__name__)


class AnswerGenerator:
    """Generate grounded answers using Gemini LLM."""
    
    def __init__(self):
        """Initialize answer generator."""
        # Load environment and API key
        load_env()
        api_key = get_api_key()
        genai.configure(api_key=api_key)
        
        # Load model configuration
        model_config = get_model_config()
        llm_config = model_config.get('llm_model', {})
        
        self.model_name = llm_config.get('name', 'gemini-1.5-flash')
        self.temperature = llm_config.get('temperature', 0.3)
        self.max_tokens = llm_config.get('max_output_tokens', 2048)
        self.top_p = llm_config.get('top_p', 0.95)
        self.top_k = llm_config.get('top_k', 40)
        
        # Initialize model
        self.model = genai.GenerativeModel(self.model_name)
        
        logger.info(f"Answer Generator initialized: {self.model_name}")
    
    def create_prompt(self, question: str, context: str) -> str:
        """
        Create a prompt for answer generation.
        
        Args:
            question: User question
            context: Retrieved context
            
        Returns:
            Formatted prompt
        """
        prompt = f"""You are a helpful AI assistant that answers questions based on provided document context.

INSTRUCTIONS:
1. Answer the question using ONLY the information from the provided context
2. If the answer is not in the context, say "I cannot answer this question based on the provided documents"
3. Include specific page citations in your answer using the format [Source X, Page Y]
4. Be detailed and comprehensive in your answer
5. Quote relevant parts of the context when appropriate
6. Do not make up information or use external knowledge

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""
        
        return prompt
    
    def extract_citations(self, retrieved_chunks: List[Dict]) -> List[Dict]:
        """
        Extract citation information from retrieved chunks.
        
        Args:
            retrieved_chunks: List of retrieved chunk dictionaries
            
        Returns:
            List of citation dictionaries
        """
        citations = []
        for i, chunk in enumerate(retrieved_chunks):
            metadata = chunk.get('metadata', {})
            citations.append({
                'source_number': i + 1,
                'document_title': metadata.get('document_title', 'Unknown'),
                'page_number': metadata.get('page_number', 'Unknown'),
                'filename': metadata.get('filename', 'Unknown'),
                'similarity': chunk.get('similarity', 0)
            })
        
        return citations
    
    def generate_answer(
        self, 
        question: str, 
        context: str,
        retrieved_chunks: List[Dict] = None
    ) -> Dict:
        """
        Generate an answer to the question using the context.
        
        Args:
            question: User question
            context: Retrieved context string
            retrieved_chunks: Original retrieved chunks for citations
            
        Returns:
            Dictionary with answer and metadata
        """
        logger.info(f"Generating answer for question: {question[:100]}...")
        
        # Create prompt
        prompt = self.create_prompt(question, context)
        
        try:
            # Generate answer
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=self.temperature,
                    max_output_tokens=self.max_tokens,
                    top_p=self.top_p,
                    top_k=self.top_k
                )
            )
            
            answer_text = response.text
            
            # Extract citations if chunks provided
            citations = []
            if retrieved_chunks:
                citations = self.extract_citations(retrieved_chunks)
            
            result = {
                'question': question,
                'answer': answer_text,
                'citations': citations,
                'num_sources': len(citations),
                'model': self.model_name
            }
            
            logger.info(f"Answer generated successfully with {len(citations)} citations")
            return result
        
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return {
                'question': question,
                'answer': f"Error generating answer: {str(e)}",
                'citations': [],
                'num_sources': 0,
                'model': self.model_name,
                'error': str(e)
            }
    
    def answer_question(
        self,
        question: str,
        retrieved_chunks: List[Dict]
    ) -> Dict:
        """
        Answer a question using retrieved chunks.
        
        Args:
            question: User question
            retrieved_chunks: List of retrieved chunk dictionaries
            
        Returns:
            Answer dictionary
        """
        # Format context from chunks
        from src.qa_pipeline.retriever import HybridRetriever
        retriever = HybridRetriever()
        context = retriever.format_context(retrieved_chunks)
        
        # Generate answer
        return self.generate_answer(question, context, retrieved_chunks)
