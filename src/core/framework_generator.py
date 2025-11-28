from src.clients.api import get_response
from src.utils.helper import extract_python_code, load_prompt


class FrameworkGenerator:
    def __init__(self, workspace_dir, model):
        self.workspace_dir = workspace_dir
        self.model = model

    def generate(self, dmte_summary, overall_summary, addendum_section, output_path):
        """
        Use LLM to generate a code framework.
        """
        prompt = load_prompt(
            "generate_framework",
            dmte_summary=dmte_summary,
            overall_summary=overall_summary,
            addendum_section=addendum_section,
        )
        while True:
            response = get_response(prompt, self.model, temp0=True)
            code_framework = extract_python_code(response, output_path, save=True)
            if code_framework.strip():
                return code_framework
