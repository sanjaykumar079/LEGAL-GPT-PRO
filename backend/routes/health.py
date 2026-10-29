from fastapi import APIRouter, Depends
from models.schemas import HealthResponse
from services.rag_service import RAGService

router = APIRouter(tags=["Health"])


def get_rag_service():
    # Import locally or use app state singleton
    from main import rag_service
    return rag_service


@router.get("/health", response_model=HealthResponse)
def health_check():
    from main import rag_service
    
    stats = rag_service.get_knowledge_base_stats()
    ollama_ok = False
    
    try:
        models = rag_service.llm_service.get_available_models()
        ollama_ok = len(models) > 0
    except Exception:
        ollama_ok = False

    return HealthResponse(
        status="healthy",
        ollama_connected=ollama_ok,
        chromadb_chunks=stats.get('total_chunks', 0)
    )