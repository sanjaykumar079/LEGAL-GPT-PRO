import pymupdf  # Changed from fitz to pymupdf
from typing import List, Dict, Any
from dataclasses import dataclass
import os

@dataclass
class PageContent:
    page_number: int
    text: str
    word_count: int
    
    def to_dict(self):
        return {
            'page': self.page_number,
            'text': self.text,
            'word_count': self.word_count
        }

class PDFService:
    def __init__(self):
        self.supported_formats = ['.pdf']
    
    def extract_text_from_pdf(self, pdf_path: str) -> List[PageContent]:
        """
        Extract text from PDF preserving page boundaries
        Returns list of PageContent objects
        """
        # Validate file exists and is PDF
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        if not pdf_path.lower().endswith('.pdf'):
            raise ValueError("File must be a PDF")
        
        try:
            doc = pymupdf.open(pdf_path)  # Changed from fitz to pymupdf
            pages = []
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text('text')  # Extract plain text
                
                # Clean text
                text = self._clean_text(text)
                
                pages.append(PageContent(
                    page_number=page_num + 1,  # 1-indexed for legal citations
                    text=text,
                    word_count=len(text.split())
                ))
            
            doc.close()
            return pages
            
        except pymupdf.FileDataError as e:  # Changed from fitz to pymupdf
            raise ValueError(f"Corrupt or encrypted PDF: {str(e)}")
        except Exception as e:
            raise Exception(f"PDF extraction failed: {str(e)}")
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize extracted text"""
        import re
        
        # Remove excessive whitespace but preserve paragraphs
        text = re.sub(r'\n\s*\n', '\n\n', text)  # Multiple newlines to double newline
        text = re.sub(r'[ \t]+', ' ', text)      # Multiple spaces/tabs to single space
        
        return text.strip()
    
    def get_document_info(self, pdf_path: str) -> Dict[str, Any]:
        """Get PDF metadata without full extraction"""
        try:
            doc = pymupdf.open(pdf_path)  # Changed from fitz to pymupdf
            info = {
                'filename': os.path.basename(pdf_path),
                'total_pages': len(doc),
                'is_encrypted': doc.is_encrypted,
                'metadata': doc.metadata
            }
            doc.close()
            return info
        except Exception as e:
            raise Exception(f"Failed to read PDF info: {str(e)}")

# Convenience function for quick testing
def extract_pdf_text(pdf_path: str) -> List[Dict]:
    """Quick function for testing PDF extraction"""
    service = PDFService()
    pages = service.extract_text_from_pdf(pdf_path)
    return [page.to_dict() for page in pages]