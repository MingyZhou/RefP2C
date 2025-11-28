import logging
import os

from src.clients.api import get_response
from src.utils.helper import extract_markdown, load_prompt


class PaperSummarizer:
    def __init__(self, workspace_dir, model="o3-mini"):
        self.workspace_dir = workspace_dir
        self.model = model

    def summarize(self, paper_content, replace=False):
        """
        Summarizes the paper content from 'paper.md' and saves it to a file.
        Logs the LLM interaction (prompt and response) to a separate file.
        """
        output_path_dmte = os.path.join(self.workspace_dir, "paper_summary_dmte.md")
        dmte_summary = ""
        if os.path.exists(output_path_dmte) and not replace:
            with open(output_path_dmte, "r", encoding="utf-8") as f:
                dmte_summary = f.read()
            logging.info(
                f"Summary already exists at {output_path_dmte}. Returning existing summary."
            )
        else:
            dmte_prompt = load_prompt(
                "summarize_dmte",
                paper_content=paper_content,
            )
            response = get_response(
                prompt=dmte_prompt,
                model=self.model,
            )
            dmte_summary = extract_markdown(
                response, file_path=output_path_dmte, save=True
            )

        output_path_workflow = os.path.join(
            self.workspace_dir, "paper_summary_workflow.md"
        )
        workflow_summary = ""
        if os.path.exists(output_path_workflow) and not replace:
            with open(output_path_workflow, "r", encoding="utf-8") as f:
                workflow_summary = f.read()
            logging.info(
                f"Summary already exists at {output_path_workflow}. Returning existing summary."
            )
        else:
            workflow_prompt = load_prompt(
                "summarize_workflow",
                paper_content=paper_content,
            )
            response = get_response(
                prompt=workflow_prompt,
                model=self.model,
            )
            workflow_summary = extract_markdown(
                response, file_path=output_path_workflow, save=True
            )

        return dmte_summary, workflow_summary
