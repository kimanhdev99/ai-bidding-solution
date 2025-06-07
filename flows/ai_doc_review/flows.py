import os
from pathlib import Path
from typing import Dict, Any

from promptflow.client import load_flow
from promptflow.connections import CustomConnection
from promptflow.entities import FlowContext
from common.models import IssueType

MODELS_MODULE_PATH = Path(__file__).parent / "common" / "models.py"
TEMPLATE_FLOW_PATH = Path(__file__).parent / "agent_template"
PROMPTS_PATH = Path(__file__).parent / "prompts"

AGENT_PROMPTS = {
   IssueType.GrammarSpelling: {
       "agent": PROMPTS_PATH / "grammar" / "agent.jinja2",
       "consolidator": PROMPTS_PATH / "grammar" / "consolidator.jinja2",
       "guidelines": PROMPTS_PATH / "grammar" / "guidelines.jinja2",
   },
    IssueType.DefinitiveLanguage: {
         "agent": PROMPTS_PATH / "definitive_language" / "agent.jinja2",
         "consolidator": PROMPTS_PATH / "definitive_language" / "consolidator.jinja2",
         "guidelines": PROMPTS_PATH / "definitive_language" / "guidelines.jinja2",
    }
}


def create_flow(agent_prompt_path, consolidator_prompt_path, guidelines_prompt_path, connection):
    flow = load_flow(TEMPLATE_FLOW_PATH)
    flow.context = FlowContext(
        connections={
            "llm_multishot": {"connection": connection},
            "consolidator": {"connection": connection},
        },
        overrides={
            "nodes.agent_prompt.source.path": str(agent_prompt_path),
            "nodes.consolidator_prompt.source.path": str(consolidator_prompt_path),
            "nodes.guidelines_prompt.source.path": str(guidelines_prompt_path),
            "nodes.llm_multishot.inputs.module_path": str(MODELS_MODULE_PATH),
            "nodes.consolidator.inputs.module_path": str(MODELS_MODULE_PATH),
            "nodes.llm_multishot.source.type": "python",
            "nodes.llm_multishot.source.path": "huggingface_client.py",
            "nodes.consolidator.source.type": "python",
            "nodes.consolidator.source.path": "huggingface_client.py"
        }
    )
    return flow


def setup_flows():
    connection = CustomConnection(
        name="hf_connection",
        configs={
            "hf_model_name": os.environ.get("HF_MODEL_NAME", "microsoft/Phi-3-mini-4k-instruct"),
            "hf_api_url": "https://api-inference.huggingface.co/models"
        },
        secrets={
            "hf_api_token": ""
        }
    )

    return {
        issue_type: create_flow(
            agent_prompt_path=AGENT_PROMPTS[issue_type]["agent"],
            consolidator_prompt_path=AGENT_PROMPTS[issue_type]["consolidator"],
            guidelines_prompt_path=AGENT_PROMPTS[issue_type]["guidelines"],
            connection=connection,
        )
        for issue_type in AGENT_PROMPTS
    }
