import logging
import os

from src.clients.api import get_response
from src.utils.helper import (extract_markdown, extract_python_code,
                              load_prompt, read_file)


class ExperimentGenerator:
    def __init__(self, workspace_dir, model):
        self.workspace_dir = workspace_dir
        self.model = model

    def generate(
        self, paper, addendum_section, code, plan_model="gpt-4o-mini", replace=False
    ):
        output_code_path = os.path.join(self.workspace_dir, "experiments.py")
        if os.path.exists(output_code_path) and not replace:
            code = read_file(output_code_path)
            logging.info(
                f"File {output_code_path} already exists. Skipping code generation."
            )
            return code, output_code_path
        plan_prompt = load_prompt(
            "generate_exp_plan",
            paper=paper,
            addendum_section=addendum_section,
            code=code,
        )
        output_plan_path = os.path.join(
            self.workspace_dir, "../../intermediates/experiment_plan.md"
        )
        plan_response = get_response(plan_prompt, model=plan_model)
        exp_plan = extract_markdown(
            plan_response, file_path=output_plan_path, save=True
        )
        exp_prompt = load_prompt(
            "generate_exp",
            paper=paper,
            code=code,
            addendum_section=addendum_section,
            exp_plan=exp_plan,
        )
        while True:
            response = get_response(exp_prompt, self.model, temp0=True)
            code = extract_python_code(response, output_code_path, save=True)
            if code.strip():
                return code, output_code_path
