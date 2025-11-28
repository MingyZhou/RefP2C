import logging
import os
import re
from typing import Dict, List, Tuple

from src.utils.helper import sanitize_code, save_code

from .editor import CodeEditor
from .revision_planner import RevisionPlanner
from .verifier import CodeVerifier


class RefinementController:
    def __init__(
        self,
        workspace_dir: str,
        verifier: CodeVerifier,
        planner: RevisionPlanner,
        editor: CodeEditor,
    ):
        self.workspace_dir = workspace_dir
        self.verifier = verifier
        self.planner = planner
        self.editor = editor

    def run_refinement_cycle(
        self,
        initial_project: Dict[str, str],
        criteria_data: List[Dict],
        paper_text: str,
        max_major_attempts: int = 3,
        log_dir_name: str = "code_reflection",
    ) -> Dict[str, str]:
        current_project = initial_project.copy()

        log_dir = os.path.join(self.workspace_dir, log_dir_name)
        os.makedirs(log_dir, exist_ok=True)
        logging.info(f"Saving all refinement artifacts to: {log_dir}")

        revision_history = []

        for attempt in range(max_major_attempts):
            logging.info(
                f"=============== Major Refinement Cycle {attempt + 1}/{max_major_attempts} ==============="
            )

            round_dir = os.path.join(log_dir, f"round_{attempt + 1}")
            os.makedirs(round_dir, exist_ok=True)

            is_ok, feedback = self.verifier.verify(
                code_project=current_project,
                criteria_data=criteria_data,
                paper_text=paper_text,
            )

            eval_log_path = os.path.join(round_dir, "evaluation_feedback.md")
            with open(eval_log_path, "w", encoding="utf-8") as f:
                f.write(
                    f"# Evaluation Feedback for Round {attempt + 1}\n\n**Result:** {'PASSED' if is_ok else 'FAILED'}\n\n---\n\n{feedback}"
                )
            logging.info(
                f"Evaluation feedback for round {attempt + 1} saved to {eval_log_path}"
            )

            if is_ok:
                logging.info(
                    "✅ Code has passed all verifications. Refinement complete."
                )
                break

            logging.info("Generating a revision plan based on failures...")
            revision_plan = self.planner.generate_plan(
                feedback=feedback,
                code_project=current_project,
            )
            plan_log_path = os.path.join(round_dir, "revision_plan.md")
            with open(plan_log_path, "w", encoding="utf-8") as f:
                f.write(revision_plan)
            logging.info(
                f"Revision plan for round {attempt + 1} saved to {plan_log_path}"
            )

            config_plan, code_plan = self._parse_dual_plan(revision_plan)

            if config_plan:
                logging.info("--- Revising config.yaml ---")
                current_project["config.yaml"] = self.editor.revise_config(
                    current_project["config.yaml"], config_plan
                )
                logging.info("config.yaml has been updated.")

            if code_plan:
                logging.info("--- Revising Python code files ---")
                py_files_to_revise = re.findall(r"##\s*Code:\s*(\S+\.py)", code_plan)

                for filename in py_files_to_revise:
                    if filename not in current_project:
                        logging.warning(
                            f"Plan mentions file '{filename}' which is not in the project. Skipping."
                        )
                        continue

                    specific_plan = self._extract_plan_for_file(code_plan, filename)
                    if not specific_plan:
                        logging.warning(
                            f"Could not extract a specific plan for '{filename}'. Skipping."
                        )
                        continue

                    logging.info(f"--- Revising file: {filename} ---")
                    revised_content, revision_history = self.editor.revise_single_file(
                        file_to_revise=filename,
                        current_project=current_project,
                        revision_plan=specific_plan,
                        revision_history=revision_history,
                        paper_text=paper_text,
                    )
                    current_project[filename] = revised_content

            logging.info(f"Saving full project snapshot for round {attempt + 1}")
            for fname, content in current_project.items():
                snapshot_path = os.path.join(round_dir, fname)
                save_code(sanitize_code(content), snapshot_path)

            if attempt == max_major_attempts - 1:
                logging.warning("Maximum refinement attempts reached.")

        logging.info("--- Refinement Cycle Finished ---")
        return current_project

    def _parse_dual_plan(self, plan: str) -> Tuple[str, str]:
        config_plan_match = re.search(
            r"###\s*CONFIG_PLAN\n(.*?)(?=\n###\s*CODE_PLAN|\Z)", plan, re.DOTALL
        )
        code_plan_match = re.search(r"###\s*CODE_PLAN\n(.*)", plan, re.DOTALL)

        config_plan = config_plan_match.group(1).strip() if config_plan_match else ""
        code_plan = code_plan_match.group(1).strip() if code_plan_match else ""

        if "no changes needed" in config_plan.lower():
            config_plan = ""
        return config_plan, code_plan

    def _extract_plan_for_file(self, code_plan: str, filename: str) -> str:
        pattern = re.compile(
            rf"##\s*Code:\s*{re.escape(filename)}\n(.*?)(?=\n##\s*Code:|\Z)", re.DOTALL
        )
        match = pattern.search(code_plan)
        return match.group(1).strip() if match else ""
