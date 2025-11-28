import concurrent.futures
import logging
import re
from typing import Any, Dict, List, Tuple

from tqdm import tqdm

from src.clients.api import get_response
from src.utils.helper import load_prompt


class CodeVerifier:
    def __init__(self, model: str, max_workers: int = 10):
        self.model = model
        self.max_workers = max_workers

    def verify(
        self, code_project: Dict[str, str], criteria_data: List[Dict], paper_text: str
    ) -> Tuple[bool, str]:
        logging.info(f"Normalizing {len(criteria_data)} criteria for verification...")
        normalized_rules = [self._normalize_criterion(rule) for rule in criteria_data]

        full_code_context = "\n\n".join(
            f"--- START OF FILE: {filename} ---\n{content}\n--- END OF FILE: {filename} ---"
            for filename, content in code_project.items()
        )

        failed_feedback_reports = []
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.max_workers
        ) as executor:
            future_to_rule = {
                executor.submit(
                    self._verify_single_criterion, full_code_context, paper_text, rule
                ): rule
                for rule in normalized_rules
            }

            progress_bar = tqdm(
                concurrent.futures.as_completed(future_to_rule),
                total=len(normalized_rules),
                desc="Verifying All Criteria",
            )

            for future in progress_bar:
                rule = future_to_rule[future]
                criterion = rule.get("criterion_text", "No criterion provided.")
                try:
                    is_met, parsed_result = future.result()
                    if not is_met:
                        failure_report = (
                            f"Criterion NOT MET: {criterion}\n"
                            f"- Score: {parsed_result.get('score', 'N/A')}\n"
                            f"- Expectations: {parsed_result.get('expectations', 'N/A')}\n"
                            f"- Reality: {parsed_result.get('reality', 'N/A')}\n"
                            f"- Reason: {parsed_result.get('reason', 'N/A')}"
                        )
                        failed_feedback_reports.append(failure_report)
                except Exception as exc:
                    logging.error(
                        f"Criterion '{criterion}' generated an exception during verification: {exc}"
                    )
                    failed_feedback_reports.append(
                        f"Criterion '{criterion}' FAILED TO VERIFY: {exc}"
                    )

        if failed_feedback_reports:
            final_feedback = (
                "The following criteria were not met:\n\n"
                + "\n\n---\n\n".join(failed_feedback_reports)
            )
            return False, final_feedback
        else:
            return True, "All verification criteria were successfully met."

    def _normalize_criterion(self, criterion_data: Dict) -> Dict:
        if "criterion" in criterion_data:
            return {"criterion_text": criterion_data["criterion"]}
        logging.warning(
            f"Unrecognized criterion format in normal mode: {criterion_data}"
        )
        return {"criterion_text": str(criterion_data)}

    def _verify_single_criterion(
        self, full_code_context: str, paper_text: str, normalized_criterion: Dict
    ) -> Tuple[bool, Dict[str, Any]]:
        system_prompt = load_prompt("verify_code_system")
        criterion_text = normalized_criterion.get(
            "criterion_text", "No criterion text provided."
        )

        user_prompt = load_prompt(
            "verify_code_user",
            paper_text=paper_text,
            full_code_context=full_code_context,
            criterion_text=criterion_text,
        )

        llm_response = get_response(
            user_prompt, model=self.model, system_prompt_extra=system_prompt
        )

        try:
            pattern = re.compile(
                r"#\s*Expectations\s*(.*?)\s*#\s*Reality\s*(.*?)\s*#\s*Score\s*(.*)",
                re.DOTALL | re.IGNORECASE,
            )
            match = pattern.search(llm_response)

            if not match:
                raise ValueError(
                    "LLM response did not contain the required # Expectations, # Reality, and # Score sections."
                )

            expectations = match.group(1).strip()
            reality = match.group(2).strip()
            score_section = match.group(3).strip()

            score_num_match = re.search(r"\b([01])\b", score_section)
            if not score_num_match:
                raise ValueError(
                    "Could not find a score of 0 or 1 in the # Score section."
                )

            score = int(score_num_match.group(1))
            reason = score_section

            parsed_result = {
                "expectations": expectations,
                "reality": reality,
                "score": score,
                "reason": reason,
            }
            return score == 1, parsed_result
        except Exception as e:
            logging.error(
                f"Error parsing LLM response for verification: {e}\nResponse: {llm_response}"
            )
            return False, {"error": f"Exception during response parsing: {e}"}
