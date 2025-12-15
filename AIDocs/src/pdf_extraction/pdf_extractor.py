"""PDF text extraction with OCR fallback."""
import os
from pathlib import Path
from typing import List, Dict, Optional
import pdfplumber
from PyPDF2 import PdfReader
from src.utils import setup_logger, get_paths, save_json, ensure_dir
from .ocr_handler import OCRHandler

logger = setup_logger(__name__)


class PDFExtractor:
    """Extract text from PDFs using pdfplumber with OCR fallback."""
    
    def __init__(self):
        """Initialize PDF extractor."""
        self.paths = get_paths()
        self.ocr_handler = OCRHandler(self.paths['temp_images_dir'])
        ensure_dir(self.paths['extracted_texts_dir'])
        logger.info("PDF Extractor initialized")
    
    def get_pdf_metadata(self, pdf_path: str) -> Dict:
        """
        Extract metadata from PDF.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Dictionary with metadata
        """
        try:
            reader = PdfReader(pdf_path)
            metadata = reader.metadata or {}
            
            # Convert metadata to serializable format
            def convert_value(val):
                """Convert PyPDF2 objects to serializable types."""
                if val is None:
                    return None
                # Convert to string to handle IndirectObject and other PyPDF2 types
                return str(val)
            
            return {
                'title': convert_value(metadata.get('/Title')) or Path(pdf_path).stem,
                'author': convert_value(metadata.get('/Author')) or 'Unknown',
                'pages': len(reader.pages),
                'filename': Path(pdf_path).name,
                'filepath': pdf_path
            }
        except Exception as e:
            logger.error(f"Error extracting metadata from {pdf_path}: {e}")
            return {
                'title': Path(pdf_path).stem,
                'author': 'Unknown',
                'pages': 0,
                'filename': Path(pdf_path).name,
                'filepath': pdf_path
            }
    
    def extract_with_pdfplumber(self, pdf_path: str) -> List[Dict]:
        """
        Extract text using pdfplumber.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            List of dictionaries with page number and text
        """
        results = []
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for i, page in enumerate(pdf.pages):
                    text = page.extract_text() or ""
                    results.append({
                        'page_number': i + 1,
                        'text': text,
                        'method': 'pdfplumber'
                    })
            
            logger.info(f"Extracted text from {len(results)} pages using pdfplumber")
            return results
        
        except Exception as e:
            logger.error(f"Error extracting with pdfplumber: {e}")
            return []
    
    def needs_ocr(self, pages: List[Dict], threshold: float = 0.1) -> bool:
        """
        Determine if PDF needs OCR based on extracted text.
        
        Args:
            pages: List of page dictionaries
            threshold: Minimum ratio of pages with text
            
        Returns:
            True if OCR is needed
        """
        if not pages:
            return True
        
        pages_with_text = sum(1 for page in pages if len(page['text'].strip()) > 50)
        ratio = pages_with_text / len(pages)
        
        needs_ocr = ratio < threshold
        if needs_ocr:
            logger.info(f"Only {ratio:.1%} of pages have text. OCR will be used.")
        
        return needs_ocr
    
    def extract_text(self, pdf_path: str, force_ocr: bool = False) -> Dict:
        """
        Extract text from PDF with automatic OCR fallback.
        
        Args:
            pdf_path: Path to PDF file
            force_ocr: Force OCR even if text extraction works
            
        Returns:
            Dictionary with metadata and extracted pages
        """
        logger.info(f"Processing PDF: {pdf_path}")
        
        # Get metadata
        metadata = self.get_pdf_metadata(pdf_path)
        
        # Try pdfplumber first
        pages = []
        if not force_ocr:
            pages = self.extract_with_pdfplumber(pdf_path)
        
        # Use OCR if needed
        if force_ocr or self.needs_ocr(pages):
            logger.info("Using OCR for text extraction")
            pages = self.ocr_handler.extract_text_from_pdf_ocr(pdf_path)
        
        # Calculate statistics
        total_chars = sum(len(page['text']) for page in pages)
        
        result = {
            'metadata': metadata,
            'pages': pages,
            'statistics': {
                'total_pages': len(pages),
                'total_characters': total_chars,
                'extraction_method': pages[0]['method'] if pages else 'none'
            }
        }
        
        logger.info(f"Extraction complete: {len(pages)} pages, {total_chars} characters")
        return result
    
    def process_pdf(self, pdf_path: str, save_output: bool = True) -> Dict:
        """
        Process a PDF file and optionally save the output.
        
        Args:
            pdf_path: Path to PDF file
            save_output: Whether to save extracted text to JSON
            
        Returns:
            Extraction result dictionary
        """
        result = self.extract_text(pdf_path)
        
        if save_output:
            output_filename = Path(pdf_path).stem + '_extracted.json'
            output_path = os.path.join(self.paths['extracted_texts_dir'], output_filename)
            save_json(result, output_path)
            logger.info(f"Saved extraction result to {output_path}")
        
        return result
    
    def process_directory(self, directory: str = None) -> List[Dict]:
        """
        Process all PDFs in a directory.
        
        Args:
            directory: Directory path (defaults to raw_pdfs_dir)
            
        Returns:
            List of extraction results
        """
        if directory is None:
            directory = self.paths['raw_pdfs_dir']
        
        pdf_files = list(Path(directory).glob('*.pdf'))
        logger.info(f"Found {len(pdf_files)} PDF files in {directory}")
        
        results = []
        for pdf_path in pdf_files:
            try:
                result = self.process_pdf(str(pdf_path))
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing {pdf_path}: {e}")
        
        logger.info(f"Processed {len(results)} PDFs successfully")
        return results
    
    def cleanup(self) -> None:
        """Clean up temporary files."""
        self.ocr_handler.cleanup()
