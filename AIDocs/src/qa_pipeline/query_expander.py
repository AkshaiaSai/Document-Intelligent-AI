"""Query expansion using Gemini LLM."""
from typing import List
import google.generativeai as genai
from src.utils import setup_logger, get_model_config, get_api_key, load_env, get_settings

logger = setup_logger(__name__)


class QueryExpander:
    """Expand queries to improve retrieval using Gemini."""
    
    def __init__(self):
        """Initialize query expander."""
        # Load environment and API key
        load_env()
        api_key = get_api_key()
        genai.configure(api_key=api_key)
        
        # Load configurations
        model_config = get_model_config()
        expansion_config = model_config.get('query_expansion_model', {})
        
        self.model_name = expansion_config.get('name', 'gemini-1.5-flash')
        self.temperature = expansion_config.get('temperature', 0.7)
        self.max_tokens = expansion_config.get('max_output_tokens', 512)
        
        # Load settings
        settings = get_settings()
        self.expansion_settings = settings.get('query_expansion', {})
        self.enabled = self.expansion_settings.get('enabled', True)
        self.min_variations = self.expansion_settings.get('min_variations', 3)
        self.max_variations = self.expansion_settings.get('max_variations', 7)
        
        # Initialize model
        self.model = genai.GenerativeModel(self.model_name)
        
        logger.info(f"Query Expander initialized: {self.model_name}")
    
    def expand_query(self, query: str, num_variations: int = None) -> List[str]:
        """
        Expand a query into multiple variations.
        
        Args:
            query: Original query
            num_variations: Number of variations to generate (default from config)
            
        Returns:
            List of query variations including the original
        """
        if not self.enabled:
            return [query]
        
        if num_variations is None:
            num_variations = self.max_variations
        
        num_variations = max(self.min_variations, min(num_variations, self.max_variations))
        
        prompt = f"""Generate {num_variations - 1} alternative phrasings of the following question. 
The alternatives should:
- Preserve the original meaning
- Use different words and sentence structures
- Include synonyms where appropriate
- Be suitable for document search

Original question: {query}

Provide only the alternative questions, one per line, without numbering or explanations."""
        
        try:
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=self.temperature,
                    max_output_tokens=self.max_tokens
                )
            )
            
            # Parse variations
            variations = [line.strip() for line in response.text.strip().split('\n') if line.strip()]
            
            # Add original query at the beginning
            all_variations = [query] + variations[:num_variations - 1]
            
            logger.info(f"Expanded query into {len(all_variations)} variations")
            return all_variations
        
        except Exception as e:
            logger.error(f"Error expanding query: {e}")
            return [query]
    
    def get_query_variations(self, query: str) -> List[str]:
        """
        Get query variations (alias for expand_query).
        
        Args:
            query: Original query
            
        Returns:
            List of query variations
        """
        return self.expand_query(query)
