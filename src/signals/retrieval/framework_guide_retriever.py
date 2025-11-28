import logging

from src.utils.helper import parse_hierarchical_guide

from .base_retriever import BaseGuideRetriever


class FrameworkGuideRetriever(BaseGuideRetriever):
    def retrieve(
        self, guide_content: str, paper_content: str, replace: bool = False
    ) -> list:
        logging.info("Parsing framework guide content...")

        guide_facts = parse_hierarchical_guide(guide_content)

        if not guide_facts:
            logging.warning("No facts were parsed from the provided guide content.")
            return []

        return self.retrieve_evidence(guide_facts, paper_content)
