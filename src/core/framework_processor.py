import json
import logging
import os

from src.clients.api import get_response
from src.utils.ast_parser import (extract_comment_steps_from_code,
                                  extract_definitions_in_order,
                                  restore_and_save_py_file)
from src.utils.helper import load_prompt, read_file


class FrameworkProcessor:
    def __init__(self, workspace_dir: str, model: str):
        self.workspace_dir = workspace_dir
        self.model = model

    def generate_supplement(self, code, comment, paper_content):
        """
        Use LLM to generate a supplement for the comment based on the func/class code, some comment (step) and checklist_text/paper_content. The supplement needs to be copied from the original paper.
        """
        prompt = load_prompt(
            "generate_supplement",
            code=code,
            comment=comment,
            paper_content=paper_content,
        )
        response = get_response(prompt, self.model)
        return response

    def process_and_enrich(
        self,
        code_framework,
        ast_save_name,
        paper_content,
        replace=False,
        add_supplement=True,
    ):
        if not add_supplement:
            save_name = ast_save_name.replace("ast", "plus_wo_sup").replace(
                ".json", ".py"
            )
        else:
            save_name = ast_save_name.replace("ast", "plus").replace(".json", ".py")
        save_path = os.path.join(self.workspace_dir, save_name)
        if os.path.exists(save_path) and not replace:
            code_framework = read_file(save_path)
            logging.info(
                f"Restored Python code already exists at {save_path}. Skipping extraction."
            )
            return code_framework
        json_save_path = os.path.join(self.workspace_dir, ast_save_name)
        definitions = extract_definitions_in_order(code_framework)
        if paper_content:
            for item in definitions:
                if item.get("name", "") == "main" or item.get("type", "") in [
                    "import",
                    "main",
                ]:
                    continue
                comment_supplements = {}
                comments = extract_comment_steps_from_code(item["code"])
                for comment in comments:
                    supplement = self.generate_supplement(
                        item["code"], comment, paper_content
                    )
                    comment_supplements[comment] = supplement
                if comment_supplements:
                    item["comment_supplements"] = comment_supplements
        with open(json_save_path, "w", encoding="utf-8") as f:
            json.dump(
                {"file": ast_save_name, "definitions": definitions},
                f,
                indent=2,
                ensure_ascii=False,
            )
            logging.info(f"Saved AST extraction to {json_save_path}")
        code_framework = restore_and_save_py_file(
            definitions, save_path, save=True, add_supplement=add_supplement
        )
        return code_framework
