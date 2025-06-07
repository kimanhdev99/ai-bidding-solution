from typing import Generator, Any, Dict
import json
import httpx
import asyncio
from connection.config import settings

class HuggingFaceClient:
    def __init__(self, model_name: str = None):
        self.model_name = model_name or settings.hf_model_name
        self.api_url = f"{settings.hf_api_url}/{self.model_name}"
        self.headers = {"Authorization": f"Bearer {settings.hf_api_token}"}
        
    async def _make_inference_request(self, payload: Dict) -> httpx.Response:
        """Make a request to the Hugging Face Inference API."""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                self.api_url,
                json=payload,
                headers=self.headers,
                timeout=None if settings.hf_wait_for_model else 30
            )
            response.raise_for_status()
            return response
            
    async def generate_stream(self, prompt: str) -> Generator[str, None, None]:
        """Generate streaming responses from the model using Inference API."""
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": settings.hf_max_length,
                "temperature": settings.hf_temperature,
                "do_sample": True,
                "return_full_text": False,
                "stream": True
            }
        }
        
        try:
            response = await self._make_inference_request(payload)
            
            # Process streaming response
            for line in response.iter_lines():
                print(line)
                if isinstance(line, bytes):
                    if line.startswith(b"data:"):
                        json_str = line.decode("utf-8").replace("data:", "").strip()
                        if json_str:
                            data = json.loads(json_str)
                            if "generated_text" in data:
                                yield data["generated_text"]
                            elif "token" in data:
                                yield data["token"]["text"]
                elif isinstance(line, str):
                    if line.startswith("data:"):
                        json_str = line.replace("data:", "").strip()
                        if json_str:
                            data = json.loads(json_str)
                            if "generated_text" in data:
                                yield data["generated_text"]
                            elif "token" in data:
                                yield data["token"]["text"]
                                        
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 503:
                yield json.dumps({"error": "Model is currently loading. Please try again in a few minutes."})
            else:
                yield json.dumps({"error": f"API request failed: {str(e)}"})
        except Exception as e:
            yield json.dumps({"error": f"An error occurred: {str(e)}"})
            
    async def process_document(self, pdf_name: str, batch_size: int = None) -> Generator[Any, None, None]:
        """Process a document and generate responses in batches using Inference API."""
        batch_size = batch_size or settings.hf_batch_size
        
        try:
            # Here you would implement the document processing logic
            # For demonstration, we'll yield a sample response
            yield json.dumps({"flow_output_streaming": {
                "status": "processing",
                "message": f"Processing document {pdf_name} with batch size {batch_size}",
                "model": self.model_name,
                "api_url": self.api_url
            }})
            
            # Simulate document processing with the Inference API
            # In a real implementation, you would:
            # 1. Read and chunk the document
            # 2. Process each chunk with the API
            # 3. Stream results back
            
        except Exception as e:
            yield json.dumps({"flow_output_streaming": {
                "status": "error",
                "message": str(e)
            }}) 