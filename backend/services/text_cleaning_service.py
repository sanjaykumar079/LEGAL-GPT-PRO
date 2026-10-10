import re
from typing import List, Dict, Any

class TextCleaningService:
    def __init__(self):
        # Common legal document patterns to clean
        self.legal_patterns = [
            # Remove excessive whitespace
            (r'\s+', ' '),
            # Fix broken words at line breaks (common in legal docs)
            (r'(\w+)-\s*\n\s*(\w+)', r'\1\2'),
            # Normalize multiple newlines
            (r'\n{3,}', '\n\n'),
            # Remove form feed characters
            (r'\f', '\n'),
        ]
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize legal document text"""
        if not text:
            return ""
        
        # Apply cleaning patterns
        for pattern, replacement in self.legal_patterns:
            text = re.sub(pattern, replacement, text)
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        # Normalize paragraph breaks
        text = re.sub(r'\n\s*\n', '\n\n', text)
        
        return text
    
    def remove_headers_footers(self, text: str) -> str:
        """Remove common legal document headers/footers"""
        # Remove page numbers like "Page 1 of 10"
        text = re.sub(r'Page \d+ of \d+', '', text)
        
        # Remove case numbers that often appear in headers
        text = re.sub(r'Case No\.:?\s*[\w-]+', '', text)
        
        return text.strip()
    
    def extract_legal_sections(self, text: str) -> List[Dict[str, Any]]:
        """Identify legal document sections"""
        sections = []
        section_patterns = [
            r'(SECTION\s+[IVX]+[.\-\s]*)',
            r'(Article\s+\d+[.\-\s]*)',
            r'(Part\s+\d+[.\-\s]*)',
            r'(^\d+\.\s+[A-Z][^.]*$)',  # Numbered sections
        ]
        
        for pattern in section_patterns:
            matches = re.finditer(pattern, text, re.MULTILINE | re.IGNORECASE)
            for match in matches:
                sections.append({
                    'title': match.group(1).strip(),
                    'start_pos': match.start(),
                    'end_pos': match.end()
                })
        
        return sections