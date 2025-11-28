import argparse
import logging
import os
import sys

from src.configs.config import LLM_MODELS
from src.configs.path_config import PAPER_DIR, RESULTS_DIR
from src.data_processing.load_data import PaperLoader
from src.signals.extraction.config_level_extractor import \
    ConfigLevelGuideExtractor
from src.signals.extraction.exhaustive_scan_extractor import \
    ExhaustiveScanGuideExtractor
from src.signals.extraction.framework_level_extractor import \
    FrameworkLevelGuideExtractor
from src.signals.filter.signal_filter import SignalFilter
from src.signals.retrieval.config_guide_retriever import ConfigGuideRetriever
from src.signals.retrieval.framework_guide_retriever import \
    FrameworkGuideRetriever
from src.signals.standardization.signal_standardizer import SignalStandardizer
from src.utils.helper import save_json

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
)


def parse_args():
    parser = argparse.ArgumentParser(description="Supervisional Signal Design Pipeline")

    parser.add_argument(
        "--workspace_dir",
        type=str,
        default="default",
        help="Directory to save all outputs for this run.",
    )
    parser.add_argument(
        "--paper_id", type=str, default="", help="The unique ID of the paper."
    )
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    parser.add_argument("--rerank_model", type=str, default="gpt-4o-mini")
    parser.add_argument(
        "--replace", action="store_true", help="Force overwrite of existing files."
    )
    return parser.parse_args()


class SignalDesignPipeline:
    def __init__(self, args):
        self.args = args
        self.workspace_dir = self._setup_workspace()
        self.result_dir = os.path.join(self.workspace_dir, "signal_design")
        self.paper_path = os.path.join(PAPER_DIR, self.args.paper_id, "paper.md")

        os.makedirs(self.result_dir, exist_ok=True)

        self.paper_loader = PaperLoader(self.paper_path)

        self.framework_guide_extractor = FrameworkLevelGuideExtractor(
            self.result_dir, LLM_MODELS[args.model]
        )
        self.config_guide_extractor = ConfigLevelGuideExtractor(
            self.result_dir, LLM_MODELS[args.model]
        )
        self.exhaustive_scan_extractor = ExhaustiveScanGuideExtractor(
            self.result_dir, LLM_MODELS[args.model]
        )

        self.framework_retriever = FrameworkGuideRetriever(
            self.result_dir, LLM_MODELS[args.rerank_model]
        )
        self.config_retriever = ConfigGuideRetriever(
            self.result_dir, LLM_MODELS[args.rerank_model]
        )

        self.standardizer = SignalStandardizer(
            self.result_dir, LLM_MODELS[args.model], LLM_MODELS[args.model]
        )
        self.filter = SignalFilter(self.result_dir, LLM_MODELS[args.model])

    def _setup_workspace(self):
        workspace_dir = os.path.join(RESULTS_DIR, self.args.workspace_dir)
        os.makedirs(workspace_dir, exist_ok=True)
        logging.info(f"🚀 Starting Signal Design. Workspace set to: {workspace_dir}")
        return workspace_dir

    def run(self):
        logging.info("--- Stage 1: Loading Data ---")
        paper_content = self.paper_loader.load()
        if not paper_content:
            logging.error(
                f"Paper content could not be loaded from {self.paper_path}. Halting."
            )
            sys.exit(1)

        logging.info("--- Stage 2: Extracting all three types of Guides ---")
        framework_guide_content = self.framework_guide_extractor.extract(
            paper_content, self.args.replace
        )
        config_guide_content = self.config_guide_extractor.extract(
            paper_content, self.args.replace
        )
        exhaustive_guide_facts = self.exhaustive_scan_extractor.extract(
            paper_content, self.args.replace
        )

        logging.info("--- Stage 3: Retrieving Evidence for Guides ---")
        enriched_framework_facts = self.framework_retriever.retrieve(
            framework_guide_content, paper_content
        )
        enriched_config_facts = self.config_retriever.retrieve(
            config_guide_content, paper_content
        )
        enriched_exhaustive_facts = [
            {
                "fact_path": [f"paragraph_{fact.get('paragraph_index')}"],
                "fact_sentence": fact.get("fact_sentence"),
                "retrieved_evidence": [{"sentence": fact.get("fact_sentence")}],
            }
            for fact in exhaustive_guide_facts
        ]

        logging.info("--- Stage 4: Standardizing all enriched facts into signals ---")
        framework_signals = self.standardizer.standardize(
            enriched_framework_facts, paper_content
        )
        config_signals = self.standardizer.standardize(
            enriched_config_facts, paper_content
        )
        exhaustive_signals = self.standardizer.standardize(
            enriched_exhaustive_facts, paper_content
        )

        logging.info("--- Stage 5: Merging and Filtering all signals ---")
        merged_unfiltered_signals = (
            framework_signals + config_signals + exhaustive_signals
        )
        logging.info("Total signals before filtering: {len(merged_unfiltered_signals)}")

        final_signals = self.filter.filter(
            signals_to_process=merged_unfiltered_signals, paper_content=paper_content
        )

        final_output_path = os.path.join(
            self.result_dir, "supervisory_signals_final.json"
        )
        save_json(final_signals, final_output_path)

        logging.info("=" * 50)
        logging.info("🎉🎉🎉 Signal Design Pipeline COMPLETED SUCCESSFULLY! 🎉🎉🎉")
        logging.info(f"Final signals generated: {len(final_signals)}")
        logging.info(f"Find all outputs in your workspace: {self.result_dir}")
        logging.info("=" * 50)


def main():
    try:
        args = parse_args()
        pipeline = SignalDesignPipeline(args)
        pipeline.run()
    except Exception as e:
        logging.error(
            f"An unexpected error occurred in the pipeline: {e}", exc_info=True
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
