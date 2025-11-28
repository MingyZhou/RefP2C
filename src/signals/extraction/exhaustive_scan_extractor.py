import json
import logging
import os

from tqdm import tqdm

from src.clients.api import get_multi_turn_response
from src.utils.helper import extract_json, load_prompt
from src.utils.md_processing import MarkdownParser


class ExhaustiveScanGuideExtractor:
    """
    Extracts a "fact-level" guide by exhaustively scanning a paper paragraph by paragraph.
    The guide is a JSONL file where each line is a verifiable factual sentence.
    """

    def __init__(self, workspace_dir: str, model: str, max_retries: int = 5):
        self.workspace_dir = workspace_dir
        self.model = model
        self.max_retries = max_retries

    def extract(self, paper_content: str, replace: bool = False) -> list:
        output_filename = "guide_exhaustive_scan.jsonl"
        output_path = os.path.join(self.workspace_dir, output_filename)

        if os.path.exists(output_path) and not replace:
            logging.info(
                f"Exhaustive scan guide already exists. Loading from: {output_path}"
            )
            facts = []
            try:
                with open(output_path, "r", encoding="utf-8") as f:
                    for line in f:
                        facts.append(json.loads(line))
                return facts
            except (json.JSONDecodeError, IOError) as e:
                logging.error(
                    f"Failed to load existing file {output_path}: {e}. Will regenerate."
                )

        system_prompt = load_prompt("extract_guide_exhaustive_scan")

        parser = MarkdownParser(paper_content)

        _, _, sentences_by_para = parser.get_parsing_results()

        logging.info(f"Loaded {len(sentences_by_para)} paragraphs for scanning.")

        conversation_history = []
        all_selected_facts = []

        with open(output_path, "w", encoding="utf-8") as f:
            for para_idx, sentences in enumerate(
                tqdm(sentences_by_para, desc="Scanning Paragraphs")
            ):
                if not sentences:
                    continue

                # Retry logic for each paragraph
                for attempt in range(self.max_retries):
                    try:
                        # Create a "draft paper" copy for this turn to ensure retries are clean
                        history_for_this_turn = conversation_history.copy()

                        formatted_sentences = [
                            f"[{i+1}]: {s}" for i, s in enumerate(sentences)
                        ]
                        user_message = (
                            "Please select the index numbers of all sentences that contain verifiable facts from the following list:\n\n"
                            + "\n".join(formatted_sentences)
                        )

                        llm_response_str = get_multi_turn_response(
                            messages=history_for_this_turn,  # Use the temporary copy
                            new_user_message=user_message,
                            system_prompt_extra=system_prompt,
                            model=self.model,
                        )

                        selected_indices = extract_json(llm_response_str)
                        if not isinstance(selected_indices, list):
                            raise TypeError("LLM response did not parse to a list.")

                        for index in selected_indices:
                            sentence_idx = int(index) - 1
                            if 0 <= sentence_idx < len(sentences):
                                fact = {
                                    "paragraph_index": para_idx,
                                    "sentence_index_in_paragraph": sentence_idx,
                                    "fact_sentence": sentences[sentence_idx],
                                }
                                f.write(json.dumps(fact, ensure_ascii=False) + "\n")
                                all_selected_facts.append(fact)
                        conversation_history = history_for_this_turn
                        break

                    except Exception as e:
                        logging.warning(
                            f"Attempt {attempt + 1}/{self.max_retries} failed for paragraph {para_idx}: {e}"
                        )
                        if attempt + 1 == self.max_retries:
                            logging.error(
                                f"All retries failed for paragraph {para_idx}. Skipping."
                            )

        logging.info(
            f"Extraction complete. Selected {len(all_selected_facts)} verbatim fact sentences. Saved to {output_path}."
        )
        return all_selected_facts
