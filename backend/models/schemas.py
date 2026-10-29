from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any


# ── Health Check ─────────────────────────────────────────────────────────────
class HealthResponse(BaseModel):
    status: str
    ollama_connected: bool
    chromadb_chunks: int


# ── Chat Schemas ─────────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    question: str = Field(
        ..., 
        example="What did the court rule regarding the defendant's motion?",
        description="The legal question to ask."
    )
    top_k: int = Field(
        default=5, 
        ge=1, 
        le=15, 
        description="Number of chunks to retrieve."
    )
    document_filter: Optional[str] = Field(
        default=None, 
        example="court_order_v2.pdf",
        description="Optional filename filter to constrain retrieval."
    )


class CitationSchema(BaseModel):
    document: str
    page: Any
    similarity: float
    preview: str


class ChatResponse(BaseModel):
    status: str = Field(..., description="ANSWERED | REFUSED_RETRIEVAL | REFUSED_LLM | ERROR")
    answer: str
    top1_similarity: float
    mean_similarity: float
    retrieved_chunks: int
    citations: List[CitationSchema]
    reason: Optional[str] = None
    error: bool = False


# ── Document Management Schemas ──────────────────────────────────────────────
class UploadResponse(BaseModel):
    filename: str
    total_pages: int
    total_chunks: int
    chunk_ids: List[str]


class DocumentInfo(BaseModel):
    filename: str
    chunk_count: int


class DocumentListResponse(BaseModel):
    total_documents: int
    total_chunks: int
    documents: List[DocumentInfo]


class DeleteDocumentResponse(BaseModel):
    filename: str
    deleted_chunks: int
    message: str