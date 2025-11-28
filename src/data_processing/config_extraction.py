import logging
import os
import shutil

from src.clients.api import get_response
from src.utils.helper import extract_yaml, load_prompt


class ConfigExtractor:
    def __init__(self, file_path, model="gpt-4o-mini"):
        self.file_path = file_path
        self.model = model

    def extract_config(self, paper, addendum_section, replace=False):
        output_path = os.path.join(self.file_path, "config.yaml")
        if os.path.exists(output_path) and not replace:
            logging.info(
                f"Summary already exists at {output_path}. Returning existing configuration."
            )
            with open(output_path, "r", encoding="utf-8") as f:
                config = f.read()
        else:
            config_prompt = load_prompt(
                "extract_config", paper=paper, addendum_section=addendum_section
            )
            llm_response = get_response(
                prompt=config_prompt,
                model=self.model,
            )

            # extract and save
            config = extract_yaml(llm_response, output_path, save=True)

        extra_path = os.path.join(self.file_path, "../repo/initial_repo/config.yaml")
        os.makedirs(os.path.dirname(extra_path), exist_ok=True)
        shutil.copyfile(output_path, extra_path)

        return config
