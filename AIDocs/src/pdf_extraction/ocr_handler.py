"""OCR handler for processing images and scanned PDFs."""
import os
from pathlib import Path
from typing import List, Optional
from PIL import Image
import pytesseract
from pdf2image import convert_from_path
from src.utils import setup_logger, get_settings, ensure_dir

logger = setup_logger(__name__)


class OCRHandler:
    """Handle OCR processing for images and scanned PDFs."""
    
    def __init__(self, temp_dir: str = "data/temp_images"):
        """
        Initialize OCR handler.
        
        Args:
            temp_dir: Directory for temporary image storage
        """
        self.temp_dir = temp_dir
        ensure_dir(temp_dir)
        
        # Load OCR settings
        settings = get_settings()
        self.ocr_config = settings.get('ocr', {})
        self.language = self.ocr_config.get('language', 'eng')
        self.dpi = self.ocr_config.get('dpi', 300)
        self.preprocessing = self.ocr_config.get('preprocessing', True)
        
        logger.info(f"OCR Handler initialized with language={self.language}, dpi={self.dpi}")
    
    def preprocess_image(self, image: Image.Image) -> Image.Image:
        """
        Preprocess image for better OCR accuracy.
        
        Args:
            image: PIL Image object
            
        Returns:
            Preprocessed image
        """
        if not self.preprocessing:
            return image
        
        # Convert to grayscale
        image = image.convert('L')
        
        # Enhance contrast (simple thresholding)
        # This can be improved with more sophisticated preprocessing
        threshold = 128
        image = image.point(lambda p: p > threshold and 255)
        
        return image
    
    def extract_text_from_image(self, image_path: str) -> str:
        """
        Extract text from a single image using OCR.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Extracted text
        """
        try:
            image = Image.open(image_path)
            
            if self.preprocessing:
                image = self.preprocess_image(image)
            
            # Perform OCR
            text = pytesseract.image_to_string(image, lang=self.language)
            
            logger.debug(f"Extracted {len(text)} characters from {image_path}")
            return text
        
        except Exception as e:
            logger.error(f"Error extracting text from image {image_path}: {e}")
            return ""
    
    def pdf_to_images(self, pdf_path: str) -> List[str]:
        """
        Convert PDF pages to images.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            List of image file paths
        """
        try:
            # Convert PDF to images
            images = convert_from_path(pdf_path, dpi=self.dpi)
            
            image_paths = []
            pdf_name = Path(pdf_path).stem
            
            for i, image in enumerate(images):
                image_path = os.path.join(self.temp_dir, f"{pdf_name}_page_{i+1}.png")
                image.save(image_path, 'PNG')
                image_paths.append(image_path)
            
            logger.info(f"Converted {len(images)} pages from {pdf_path} to images")
            return image_paths
        
        except Exception as e:
            logger.error(f"Error converting PDF to images: {e}")
            return []
    
    def extract_text_from_pdf_ocr(self, pdf_path: str) -> List[dict]:
        """
        Extract text from PDF using OCR (for scanned PDFs).
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            List of dictionaries with page number and text
        """
        # Convert PDF to images
        image_paths = self.pdf_to_images(pdf_path)
        
        if not image_paths:
            return []
        
        # Extract text from each image
        results = []
        for i, image_path in enumerate(image_paths):
            text = self.extract_text_from_image(image_path)
            results.append({
                'page_number': i + 1,
                'text': text,
                'method': 'ocr'
            })
            
            # Clean up temporary image
            try:
                os.remove(image_path)
            except Exception as e:
                logger.warning(f"Could not remove temporary image {image_path}: {e}")
        
        logger.info(f"OCR extraction completed for {pdf_path}: {len(results)} pages")
        return results
    
    def cleanup(self) -> None:
        """Clean up temporary files."""
        try:
            for file in os.listdir(self.temp_dir):
                file_path = os.path.join(self.temp_dir, file)
                if os.path.isfile(file_path):
                    os.remove(file_path)
            logger.info("Temporary OCR files cleaned up")
        except Exception as e:
            logger.error(f"Error cleaning up temporary files: {e}")
