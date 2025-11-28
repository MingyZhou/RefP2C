from src.clients.api import get_multi_turn_response
from src.utils.helper import extract_python_code, load_prompt


class StepGenerator:
    def __init__(self, workspace_dir, model):
        self.workspace_dir = workspace_dir
        self.model = model

    def generate(
        self,
        overall_summary,
        code_framework,
        addendum_section,
        info,
        conversation_history=[],
    ):
        system_prompt = load_prompt(
            "generate_steps_system",
            overall_summary=overall_summary,
            code_framework=code_framework,
            addendum_section=addendum_section,
        )
        if "summary" not in info:
            user_prompt = load_prompt(
                "generate_steps_user_simple",
                code=info["code"],
            )
        else:
            user_prompt = load_prompt(
                "generate_steps_user_detailed",
                code=info["code"],
                summary=info["summary"],
                config=info["config"],
            )
        llm_response = get_multi_turn_response(
            messages=conversation_history,
            new_user_message=user_prompt,
            model=self.model,
            system_prompt_extra=system_prompt,
        )
        part_code_new = extract_python_code(llm_response, save=False)
        return part_code_new
