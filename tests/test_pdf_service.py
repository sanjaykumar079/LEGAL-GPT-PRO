import pytest
import os
from backend.services.pdf_service import PDFService, PageContent

class TestPDFService:
    
    def setup_method(self):
        self.pdf_service = PDFService()
        self.test_pdf_path = "tests/fixtures/test_legal_doc.pdf"
    
    def test_extract_text_from_valid_pdf(self):
        """Test successful PDF text extraction"""
        if not os.path.exists(self.test_pdf_path):
            pytest.skip("Test PDF file not found")
        
        result = self.pdf_service.extract_text_from_pdf(self.test_pdf_path)
        
        # Should return list of PageContent objects
        assert isinstance(result, list)
        assert len(result) > 0
        
        # Each item should be PageContent
        for page in result:
            assert isinstance(page, PageContent)
            assert page.page_number > 0
            assert len(page.text) > 0
    
    def test_page_content_structure(self):
        """Test PageContent object structure"""
        if not os.path.exists(self.test_pdf_path):
            pytest.skip("Test PDF file not found")
        
        result = self.pdf_service.extract_text_from_pdf(self.test_pdf_path)
        first_page = result[0]
        
        # Check all required attributes exist
        assert hasattr(first_page, 'page_number')
        assert hasattr(first_page, 'text')
        assert hasattr(first_page, 'word_count')
        
        # Check data types
        assert isinstance(first_page.page_number, int)
        assert isinstance(first_page.text, str)
        assert isinstance(first_page.word_count, int)
    
    def test_page_numbering_starts_at_one(self):
        """Legal documents should have 1-indexed page numbers"""
        if not os.path.exists(self.test_pdf_path):
            pytest.skip("Test PDF file not found")
        
        result = self.pdf_service.extract_text_from_pdf(self.test_pdf_path)
        
        # First page should be page 1 (not 0)
        assert result[0].page_number == 1
    
    def test_invalid_file_raises_error(self):
        """Test error handling for invalid files"""
        with pytest.raises(FileNotFoundError):
            self.pdf_service.extract_text_from_pdf("nonexistent.pdf")
    
    def test_non_pdf_file_raises_error(self):
        """Test error handling for non-PDF files"""
        # Create a test txt file
        with open("tests/fixtures/test.txt", "w") as f:
            f.write("test content")
        
        with pytest.raises(ValueError):
            self.pdf_service.extract_text_from_pdf("tests/fixtures/test.txt")