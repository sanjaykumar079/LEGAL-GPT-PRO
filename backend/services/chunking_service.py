from typing import List, Dict, Any
import hashlib
import sys
import os

# Add backend directory to path to fix imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from services.text_cleaning_service import TextCleaningService

class ChunkingService:
    def __init__(self, chunk_size: int = 1200, overlap: int = 200):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.cleaning_service = TextCleaningService()
    
    def chunk_text(self, text: str, document: str, page: int) -> List[Dict[str, Any]]:
        """
        Split text into overlapping chunks with metadata
        
        Args:
            text: The text to chunk
            document: Document filename
            page: Page number
            
        Returns:
            List of chunk dictionaries with metadata
        """
        if not text or len(text.strip()) == 0:
            return []
        
        # Clean the text first
        cleaned_text = self.cleaning_service.clean_text(text)
        
        chunks = []
        start = 0
        chunk_index = 0
        
        while start < len(cleaned_text):
            # Calculate end position
            end = min(start + self.chunk_size, len(cleaned_text))
            
            # Try to break at sentence boundary
            if end < len(cleaned_text):
                # Look for sentence break within the last 200 characters
                search_start = max(end - 200, start)
                sentence_break = self._find_sentence_break(cleaned_text, search_start, end)
                
                if sentence_break:
                    end = sentence_break
            
            # Extract chunk text
            chunk_text = cleaned_text[start:end].strip()
            
            if chunk_text:  # Only add non-empty chunks
                chunk_id = self._generate_chunk_id(document, page, chunk_index)
                
                chunk = {
                    'text': chunk_text,
                    'document': document,
                    'page': page,
                    'chunk_id': chunk_id,
                    'chunk_index': chunk_index,
                    'start_char': start,
                    'end_char': end,
                    'char_count': len(chunk_text)
                }
                chunks.append(chunk)
            
            # Move to next chunk with overlap
            if end >= len(cleaned_text):
                break
                
            start = end - self.overlap
            chunk_index += 1
        
        return chunks
    
    def chunk_document_pages(self, pages: List[Dict[str, Any]], document: str) -> List[Dict[str, Any]]:
        """
        Chunk multiple pages from a document
        
        Args:
            pages: List of page dictionaries with 'page' and 'text' keys
            document: Document filename
            
        Returns:
            List of all chunks from all pages
        """
        all_chunks = []
        
        for page_data in pages:
            page_num = page_data.get('page', 1)
            page_text = page_data.get('text', '')
            
            page_chunks = self.chunk_text(page_text, document, page_num)
            all_chunks.extend(page_chunks)
        
        return all_chunks
    
    def _find_sentence_break(self, text: str, start: int, end: int) -> int:
        """Find the best sentence break position within range"""
        # Look for sentence endings (. ! ?) followed by space and capital letter
        for i in range(end, start, -1):
            if i > 0 and text[i-1] in '.!?':
                if i < len(text) and text[i].isspace():
                    return i
        return None
    
    def _generate_chunk_id(self, document: str, page: int, chunk_index: int) -> str:
        """Generate unique chunk identifier"""
        # Create a hash for uniqueness
        base = f"{document}_page{page}_chunk{chunk_index}"
        hash_suffix = hashlib.md5(base.encode()).hexdigest()[:6]
        return f"{base}_{hash_suffix}"
    
    def get_chunk_stats(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get statistics about the chunks"""
        if not chunks:
            return {}
        
        total_chars = sum(chunk['char_count'] for chunk in chunks)
        avg_chars = total_chars / len(chunks)
        
        return {
            'total_chunks': len(chunks),
            'total_characters': total_chars,
            'average_chunk_size': round(avg_chars, 1),
            'min_chunk_size': min(chunk['char_count'] for chunk in chunks),
            'max_chunk_size': max(chunk['char_count'] for chunk in chunks)
        }