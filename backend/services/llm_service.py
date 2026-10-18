import requests
import json
from typing import List, Dict, Any, Optional

class LLMService:
    def __init__(self, model_name: str = "llama3:latest", base_url: str = "http://localhost:11434"):
        """
        Initialize LLM service for Ollama
        
        Args:
            model_name: Name of the Ollama model to use
            base_url: Ollama API base URL
        """
        self.model_name = model_name
        self.base_url = base_url
        self._check_ollama_connection()
    
    def _check_ollama_connection(self):
        """Check if Ollama is running"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = [m['name'] for m in response.json().get('models', [])]
                print(f"✅ Ollama connected. Available models: {models}")
            else:
                print(f"⚠️ Ollama responded with status: {response.status_code}")
        except requests.exceptions.ConnectionError:
            print("❌ Cannot connect to Ollama. Ensure Ollama service is running on http://localhost:11434")
            raise Exception("Ollama not running. Start with: ollama serve")
        except Exception as e:
            print(f"⚠️ Ollama connection check failed: {e}")
    
    def generate_response(self, prompt: str, context: str = "", 
                         temperature: float = 0.1, max_tokens: int = 1000) -> str:
        """
        Generate response from LLM using Ollama
        """
        system_prompt = (
            "You are LegalMind, an expert AI legal research assistant.\n"
            "Answer the question using strictly the provided context.\n"
            "If the context does not contain sufficient factual details to answer, state clearly:\n"
            "'The provided legal context does not contain sufficient information to answer this question.'\n"
            "Do NOT assume, extrapolate, or invent legal facts or external citations."
        )
        
        full_prompt = f"""{system_prompt}

### LEGAL CONTEXT:
{context}

### USER QUESTION:
{prompt}

### ANSWER:"""
        
        try:
            data = {
                "model": self.model_name,
                "prompt": full_prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens
                }
            }
            
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=data,
                timeout=60,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get('response', 'No response generated.').strip()
            else:
                raise Exception(f"Ollama API error {response.status_code}: {response.text}")
                
        except Exception as e:
            raise Exception(f"LLM generation failed: {str(e)}")
    
    def answer_question_with_context(self, question: str, context_chunks: List[Dict[str, Any]]) -> str:
        """
        Format retrieved chunks and request answer from LLM
        """
        if not context_chunks:
            return "The provided legal context does not contain sufficient information to answer this question."
        
        context_parts = []
        for i, chunk in enumerate(context_chunks, 1):
            doc = chunk['metadata'].get('document', 'Unknown Document')
            page = chunk['metadata'].get('page', 'Unknown Page')
            context_parts.append(f"[Ref {i}] Document: {doc} | Page: {page}\n{chunk['text']}")
        
        context_str = "\n\n".join(context_parts)
        return self.generate_response(question, context=context_str)