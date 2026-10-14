import requests
import json
from typing import List, Dict, Any, Optional
import os

class LLMService:
    def __init__(self, model_name: str = "llama2", base_url: str = "http://localhost:11434"):
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
                print(f"✅ Ollama connection successful. Available models: {[m['name'] for m in response.json().get('models', [])]}")
            else:
                print(f"⚠️  Ollama responded with status: {response.status_code}")
        except requests.exceptions.ConnectionError:
            print("❌ Cannot connect to Ollama. Make sure Ollama is running on localhost:11434")
            raise Exception("Ollama not running. Start with: ollama serve")
        except Exception as e:
            print(f"⚠️  Ollama connection check failed: {e}")
    
    def generate_response(self, prompt: str, context: str = "", 
                         temperature: float = 0.1, max_tokens: int = 1000) -> str:
        """
        Generate response from LLM using Ollama
        
        Args:
            prompt: The main prompt/question
            context: Additional context to include
            temperature: Sampling temperature (0-1)
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated response text
        """
        # Construct full prompt with system instructions
        system_prompt = """You are a legal research assistant. 
        Answer using only the supplied context. 
        If the context does not contain enough information, say so. 
        Do not invent facts."""
        
        full_prompt = f"""{system_prompt}

Context:
{context}

Question: {prompt}

Answer:"""
        
        try:
            # Prepare request data
            data = {
                "model": self.model_name,
                "prompt": full_prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens
                }
            }
            
            # Make request to Ollama
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=data,
                timeout=60,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get('response', 'No response generated').strip()
            else:
                raise Exception(f"Ollama API error: {response.status_code} - {response.text}")
                
        except requests.exceptions.Timeout:
            raise Exception("Request to Ollama timed out")
        except requests.exceptions.ConnectionError:
            raise Exception("Cannot connect to Ollama. Make sure it's running.")
        except Exception as e:
            raise Exception(f"LLM generation failed: {str(e)}")
    
    def answer_question_with_context(self, question: str, context_chunks: List[Dict[str, Any]]) -> str:
        """
        Answer a question using provided context chunks
        
        Args:
            question: The question to answer
            context_chunks: List of relevant context chunks with text and metadata
            
        Returns:
            Generated answer
        """
        if not context_chunks:
            return "I don't have enough information to answer this question based on the provided documents."
        
        # Build context from chunks
        context_parts = []
        for i, chunk in enumerate(context_chunks, 1):
            source_info = f"Source: {chunk['metadata']['document']}, Page {chunk['metadata']['page']}"
            context_parts.append(f"[Context {i}] {source_info}\n{chunk['text']}")
        
        context = "\n\n".join(context_parts)
        
        # Generate answer
        return self.generate_response(question, context)
    
    def get_available_models(self) -> List[str]:
        """Get list of available Ollama models"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                return [model['name'] for model in models]
            return []
        except Exception as e:
            print(f"Failed to get models: {e}")
            return []

# Convenience function for quick testing
def create_llm_service(model_name: str = "llama2") -> LLMService:
    """Quick function to create an LLM service"""
    return LLMService(model_name=model_name)