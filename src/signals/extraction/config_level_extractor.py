import logging
import os

from src.clients.api import get_response
from src.utils.helper import extract_yaml, load_prompt, read_file


class ConfigLevelGuideExtractor:
    def __init__(self, workspace_dir: str, model: str):
        self.workspace_dir = workspace_dir
        self.model = model

    def extract(self, paper_content: str, replace: bool = False) -> str:
        output_filename = "guide_config_level.yaml"
        output_path = os.path.join(self.workspace_dir, output_filename)

        if os.path.exists(output_path) and not replace:
            logging.info(
                f"Config-level guide already exists. Loading from: {output_path}"
            )
            return read_file(output_path)

        logging.info("Generating config-level guide (verbatim config)...")

        system_prompt = load_prompt("extract_guide_config_level")

        user_prompt = (
            f"Here is the full paper content:\n```markdown\n{paper_content}\n```"
        )

        llm_response = get_response(
            prompt=user_prompt, model=self.model, system_prompt_extra=system_prompt
        )

        config_guide_content = extract_yaml(
            llm_response, file_path=output_path, save=True
        )

        return config_guide_content
