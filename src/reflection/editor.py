import logging
from typing import Dict, List, Tuple

from src.clients.api import get_multi_turn_response, get_response
from src.utils.helper import load_prompt


class CodeEditor:
    def __init__(self, workspace_dir: str, model: str):
        self.workspace_dir = workspace_dir
        self.model = model

    def revise_config(self, current_config: str, config_plan: str) -> str:
        logging.info("Revising config.yaml...")
        prompt = load_prompt(
            "revise_config_user", current_config=current_config, config_plan=config_plan
        )
        revised_config_raw = get_response(prompt, model=self.model)

        revised_config = (
            revised_config_raw.strip().replace("## config.yaml", "").strip()
        )

        return revised_config.strip()

    def revise_single_file(
        self,
        file_to_revise: str,
        current_project: Dict[str, str],
        revision_plan: str,
        revision_history: List[Dict],
        paper_text: str,
        max_retries: int = 3,
    ) -> Tuple[str, List[Dict]]:
        logging.info(f"Constructing focused prompt to revise file: {file_to_revise}")

        system_prompt = load_prompt("revise_single_file_system")
        config_yaml_content = current_project.get(
            "config.yaml", "# config.yaml not found in project context."
        )
        code_to_revise_content = current_project[file_to_revise]
        if not revision_history:
            logging.info("First turn in conversation. Including full paper text.")
            user_prompt = f"""
To ensure your revision is accurate, here is the full research paper. Refer to it if the plan is ambiguous.
<paper>
{paper_text}
</paper>

Here is the step-by-step action plan you must follow:
<revision_plan>
{revision_plan}
</revision_plan>

Here is the read-only configuration file for context.
<config_file>
{config_yaml_content}
</config_file>

Now, please fix the following file based on the plan.
<file_to_fix name="{file_to_revise}">
{code_to_revise_content}
</file_to_fix>

Remember, your output must be ONLY the complete file for `{file_to_revise}`.
"""
        else:
            logging.info("Subsequent turn. Omitting full paper text.")
            user_prompt = f"""
Here is the step-by-step action plan you must follow:
<revision_plan>
{revision_plan}
</revision_plan>

Here is the read-only configuration file for context.
<config_file>
{config_yaml_content}
</config_file>

Now, please fix the following file based on the plan.
<file_to_fix name="{file_to_revise}">
{code_to_revise_content}
</file_to_fix>

Remember, your output must be ONLY the complete file for `{file_to_revise}`.
"""

        for attempt in range(max_retries):
            logging.info(
                f"Calling LLM to revise {file_to_revise}... (Attempt {attempt + 1}/{max_retries})"
            )
            revised_file = get_multi_turn_response(
                messages=revision_history,
                new_user_message=user_prompt,
                model=self.model,
                system_prompt_extra=system_prompt,
                temp0=True,
            )
            if revised_file and revised_file.strip():
                logging.info(
                    f"Successfully received revised code for {file_to_revise}."
                )
                return revised_file, revision_history

        logging.error(
            f"Revision of {file_to_revise} failed after all retries. Returning original content."
        )
        return code_to_revise_content, revision_history
