import json
from typing import Any, AsyncGenerator
from fastapi import HTTPException
from http import HTTPStatus
from common.logger import get_logger
from flows.ai_doc_review.huggingface_client import HuggingFaceClient

logging = get_logger(__name__)

class AMLClient:
    def __init__(self, model_name: str = "mistralai/Mistral-7B-Instruct-v0.2"):
        self.hf_client = HuggingFaceClient(model_name)

    async def call_aml_endpoint(self, endpoint_name: str, pdf_name: str) -> AsyncGenerator[Any, Any]:
        """
        Process documents using Hugging Face models instead of Azure ML endpoints.

        Args:
            endpoint_name (str): Ignored (kept for compatibility)
            pdf_name (str): The name of the PDF file to process
        """
        try:
            logging.info(f"Processing document {pdf_name} with Hugging Face model...")
            async for chunk in self.hf_client.process_document(pdf_name):
                yield json.loads(chunk)
                
        except Exception as e:
            logging.error(f"An error occurred while processing the document: {e}")
            raise HTTPException(
                status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
                detail=f"Error processing document: {str(e)}"
            )
