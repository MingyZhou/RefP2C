import logging
import os

from src.utils.helper import \
    parse_verbatim_config_guide  # <-- Import our new parser

from .base_retriever import BaseGuideRetriever


class ConfigGuideRetriever(BaseGuideRetriever):
    """
    A specialist retriever for configuration-level guides.
    It inherits the main retrieval logic from BaseGuideRetriever and adds
    its own logic for parsing the input YAML guide.
    """

    def retrieve(
        self, guide_content: str, paper_content: str, replace: bool = False
    ) -> list:
        """
        The main public method for this class. It first parses the specific
        YAML guide format and then calls the generic evidence retrieval workflow.

        Args:
            guide_content (str): Path to the configuration-level guide (.yaml file).
            paper_content (str): Path to the original paper's markdown file.
            replace (bool): Whether to overwrite existing output files.

        Returns:
            list: The list of enriched facts, with evidence from the paper.
        """
        # 1. Use the dedicated helper to parse the YAML guide into facts
        guide_facts = parse_verbatim_config_guide(guide_content)

        if not guide_facts:
            logging.warning(f"No facts were parsed from guide file: {guide_content}")
            return []

        return self.retrieve_evidence(guide_facts, paper_content)
