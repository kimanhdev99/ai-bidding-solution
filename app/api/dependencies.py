from services.aml_client import AMLClient
from database.issues_repository import IssuesRepository
from services.issues_service import IssuesService
from config.config import settings


def get_issues_service() -> IssuesService:
    return IssuesService(IssuesRepository(), get_aml_client())

def get_aml_client():
    # Initialize with Hugging Face model
    return AMLClient(model_name="mistralai/Mistral-7B-Instruct-v0.2")
