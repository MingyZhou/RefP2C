import logging

from tqdm import tqdm

from src.clients.api import get_response
from src.utils.helper import (extract_json_object, load_prompt,
                              parse_json_list_from_string)


class SignalStandardizer:
    def __init__(
        self, workspace_dir: str, model: str, referee_model: str, max_retries: int = 3
    ):
        self.workspace_dir = workspace_dir
        self.model = model
        self.referee_model = referee_model
        self.max_retries = max_retries

    def standardize(self, enriched_facts: list, paper_content: str) -> list:
        logging.info(
            f"Starting signal standardization process for {len(enriched_facts)} facts..."
        )
        final_results = []
        for fact_item in tqdm(enriched_facts, desc="Standardizing Signals"):
            guide_fact = fact_item.get("fact_sentence", "")
            retrieved_evidence = fact_item.get("retrieved_evidence", [])

            if not guide_fact:
                continue

            reference_sentences_str = "\n".join(
                [item["sentence"] for item in retrieved_evidence]
            )

            initial_criteria = self._generate_initial_criteria(
                guide_fact, reference_sentences_str, paper_content
            )

            final_criteria = self._refine_criteria(
                guide_fact, reference_sentences_str, initial_criteria
            )

            if final_criteria:
                for criterion_obj in final_criteria:
                    result_item = {
                        "source_fact": guide_fact,
                        "retrieved_evidence": retrieved_evidence,
                        "criterion": criterion_obj.get(
                            "criterion", "ERROR: Criterion key missing."
                        ),
                    }
                    final_results.append(result_item)
            else:
                logging.warning(
                    f"Could not produce any valid criteria for fact: '{guide_fact[:50]}...'"
                )

        logging.info(
            f"Performing initial validation filtering on {len(final_results)} generated signals..."
        )

        final_valid_signals = []
        for item in final_results:
            criterion = item.get("criterion", "")
            if criterion and "<fact>" in criterion and "ERROR:" not in criterion:
                final_valid_signals.append(item)

        logging.info(
            f"Standardization complete. Produced {len(final_valid_signals)} valid signals."
        )
        return final_valid_signals

    def _generate_initial_criteria(
        self, guide_fact: str, reference_sentence: str, full_paper_text: str
    ) -> list:
        for attempt in range(self.max_retries):
            try:
                user_prompt = load_prompt(
                    "standardization",
                    guide_fact=guide_fact,
                    reference_sentence=reference_sentence,
                    full_paper_text=full_paper_text,
                )
                response = get_response(user_prompt, model=self.model)
                criteria = parse_json_list_from_string(response)
                if isinstance(criteria, list):
                    return criteria
            except Exception as e:
                logging.warning(
                    f"Criteria generation failed on attempt {attempt + 1}: {e}"
                )

        logging.error(
            f"Failed to generate initial criteria after {self.max_retries} attempts."
        )
        return []

    def _refine_criteria(
        self, guide_fact: str, reference_sentence: str, initial_criteria: list
    ) -> list:
        if not initial_criteria or len(initial_criteria) <= 5:
            return initial_criteria

        logging.info(
            f"Initial list has {len(initial_criteria)} criteria. Calling referee model..."
        )
        system_prompt = load_prompt("refine_standardization_system")

        formatted_criteria = "\n".join(
            f"{i}: {item.get('criterion', 'N/A')}"
            for i, item in enumerate(initial_criteria)
        )
        user_prompt = f'**Source "Guide Fact":**\n{guide_fact}\n\n**Reference Sentence (for context):**\n{reference_sentence}\n\n**Generated Criteria List (Indices 0 to {len(initial_criteria) - 1}):**\n{formatted_criteria}\n\nProvide your response in the required JSON format.'

        try:
            response_text = get_response(
                user_prompt, model=self.referee_model, system_prompt_extra=system_prompt
            )
            llm_json = extract_json_object(response_text)
            action = llm_json.get("action")

            if action == "REFINE_TO_TOP_5":
                indices = llm_json.get("indices_to_keep", [])
                logging.info(
                    f"Refinement action: REFINE_TO_TOP_5. Keeping indices: {indices[:5]}"
                )
                return [
                    initial_criteria[i]
                    for i in indices[:5]
                    if 0 <= i < len(initial_criteria)
                ]

            logging.info("Refinement action: KEEP_ORIGINAL_LIST or fallback.")
            return initial_criteria
        except Exception as e:
            logging.error(
                f"Error during criteria refinement: {e}. Keeping original list as fallback."
            )
            return initial_criteria
