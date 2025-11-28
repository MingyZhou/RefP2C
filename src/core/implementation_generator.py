import logging
import os

from src.clients.api import get_multi_turn_response
from src.utils.helper import extract_python_code, load_prompt, read_file


class ImplementationGenerator:
    def __init__(self, workspace_dir, model):
        self.workspace_dir = workspace_dir
        self.model = model

    # generate code block by block (e.g. Data, Model, Training, Evaluation)
    def generate(
        self,
        paper_content,
        code_framework,
        addendum_section,
        config,
        output_path,
        info,
        import_statements,
        conversation_history=[],
        interaction_log=[],
        replace=False,
    ):
        if os.path.exists(output_path) and not replace:
            code = read_file(output_path)
            logging.info(
                f"File {output_path} already exists. Skipping code generation."
            )
            return code, output_path

        system_prompt = load_prompt(
            "generate_code_system",
            paper_content=paper_content,
            addendum_section=addendum_section,
            config=config,
            code_framework=code_framework,
        )
        user_prompt = load_prompt(
            "generate_code_user",
            import_statements=import_statements,
            code=info["code"],
        )
        llm_response = get_multi_turn_response(
            messages=conversation_history,
            new_user_message=user_prompt,
            model=self.model,
            system_prompt_extra=system_prompt,
        )
        if len(interaction_log) < 2:
            interaction_log.append(f"System Message:\n{system_prompt}\n")
            interaction_log.append("-------------------------------------------\n")
        interaction_log.append(f"\n--- Turn {(len(interaction_log) - 3) // 3} ---\n")
        interaction_log.append(
            f"\nUser (Subsection {(len(interaction_log) - 3) // 3}):\n{user_prompt}\n"
        )
        interaction_log.append(f"\nAssistant:\n{llm_response}\n")

        new_code = extract_python_code(llm_response, save=False)
        return new_code
