import logging
from typing import Dict

from src.clients.api import get_response
from src.utils.helper import load_prompt


class RevisionPlanner:
    def __init__(self, workspace_dir: str, model: str):
        self.workspace_dir = workspace_dir
        self.model = model

    def generate_plan(self, feedback: str, code_project: Dict[str, str]) -> str:
        logging.info("Generating a structured, per-file revision plan from feedback...")

        code_context_str = ""
        for filename, content in code_project.items():
            code_context_str += f"--- START OF FILE: {filename} ---\n{content}\n--- END OF FILE: {filename} ---\n\n"

        prompt = load_prompt(
            "create_revision_plan", feedback=feedback, code_context_str=code_context_str
        )

        plan = get_response(prompt, model=self.model)

        logging.info("Revision plan generated successfully.")
        return plan.strip()
