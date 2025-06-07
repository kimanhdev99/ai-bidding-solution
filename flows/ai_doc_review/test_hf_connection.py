import asyncio
import os
from huggingface_client import HuggingFaceClient
from promptflow.connections import CustomConnection

async def test_connection():
    # Create the connection same way as in flows.py
    connection = CustomConnection(
        name="hf_connection",
        configs={
            "hf_model_name": "microsoft/Phi-3-mini-4k-instruct",
            "hf_api_url": "https://api-inference.huggingface.co/models"
        },
        secrets={
            "hf_api_token": ""
        }
    )
    
    # Initialize the client
    client = HuggingFaceClient(model_name=connection.configs["hf_model_name"])
    
    # Test prompt
    test_prompt = """
    System: You are a helpful AI assistant.
    
    User: Write a short greeting in one sentence.
    """
    
    print(f"Testing connection to model: {connection.configs['hf_model_name']}")
    print("Sending test prompt...")
    
    try:
        # Test generation
        async for response in client.generate_stream(test_prompt):
            print(f"Response: {response}")
            
        # Test document processing
        print("\nTesting document processing...")
        async for chunk in client.process_document("sample.pdf"):
            print(f"Document processing response: {chunk}")
            
    except Exception as e:
        print(f"Error during test: {str(e)}")

if __name__ == "__main__":
    # Run the async test
    asyncio.run(test_connection()) 