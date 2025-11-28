import argparse
import logging
import os
import sys

import yaml

from src.configs.config import LLM_MODELS
from src.configs.path_config import PAPER_DIR, RESULTS_DIR
from src.core.experiment_generator import ExperimentGenerator
from src.core.framework_generator import FrameworkGenerator
from src.core.framework_processor import FrameworkProcessor
from src.core.implementation_generator import ImplementationGenerator
from src.core.step_generator import StepGenerator
from src.data_processing.config_extraction import ConfigExtractor
from src.data_processing.load_data import PaperLoader
from src.data_processing.paper_summary import PaperSummarizer
from src.utils.ast_parser import (extract_definitions_in_order,
                                  extract_imports, restore_and_save_py_file)
from src.utils.helper import parse_dmte_summary, read_file

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
)


def parse_args():
    parser = argparse.ArgumentParser(description="Initial Code Implementation Pipeline")
    parser.add_argument("--paper_id", type=str, default="")
    parser.add_argument(
        "--workspace_dir",
        type=str,
        default="default",
        help="Directory to save all outputs for this run.",
    )
    parser.add_argument(
        "--model", type=str, default="gpt-4o-mini", choices=LLM_MODELS.keys()
    )
    parser.add_argument(
        "--replace", action="store_true", help="Force overwrite of existing files."
    )
    return parser.parse_args()


class CodeGenerationPipeline:
    def __init__(self, args):
        self.args = args
        self._setup_workspace()
        self.paper_path = os.path.join(PAPER_DIR, self.args.paper_id, "paper.md")

        self.paper_loader = PaperLoader(self.paper_path)
        self.paper_summarizer = PaperSummarizer(
            self.intermediates_dir, LLM_MODELS[args.model]
        )
        self.config_extractor = ConfigExtractor(
            self.intermediates_dir, LLM_MODELS[args.model]
        )
        self.framework_processor = FrameworkProcessor(
            self.intermediates_dir, LLM_MODELS[args.model]
        )
        self.framework_gen = FrameworkGenerator(
            self.intermediates_dir, LLM_MODELS[args.model]
        )
        self.step_gen = StepGenerator(self.intermediates_dir, LLM_MODELS[args.model])
        self.impl_gen = ImplementationGenerator(
            self.initial_repo_dir, LLM_MODELS[args.model]
        )
        self.exp_gen = ExperimentGenerator(
            self.initial_repo_dir, LLM_MODELS[args.model]
        )

    def _setup_workspace(self):
        self.workspace_dir = os.path.join(RESULTS_DIR, self.args.workspace_dir)
        self.intermediates_dir = os.path.join(self.workspace_dir, "intermediates")
        self.repo_dir = os.path.join(self.workspace_dir, "repo")
        self.initial_repo_dir = os.path.join(self.repo_dir, "initial_repo")

        os.makedirs(self.workspace_dir, exist_ok=True)
        os.makedirs(self.repo_dir, exist_ok=True)
        os.makedirs(self.initial_repo_dir, exist_ok=True)
        os.makedirs(self.intermediates_dir, exist_ok=True)

        logging.info(f"🚀 Starting Code Generation. Workspace: {self.workspace_dir}")

    def run(self):
        logging.info("--- Stage 1: Loading Data ---")
        paper_content = self.paper_loader.load()
        if not paper_content:
            logging.error("Paper content is empty. Halting.")
            sys.exit(1)

        paper_directory = os.path.dirname(self.paper_path)
        addendum_path = os.path.join(paper_directory, "addendum.md")

        if os.path.exists(addendum_path):
            addendum = read_file(addendum_path)
        else:
            logging.warning(
                f"Addendum file not found at {addendum_path}. Proceeding without it."
            )
            addendum = ""

        addendum_section = (
            f"\nHere is the supplementary information for code reproduction:\n{addendum}"
            if addendum.strip()
            else ""
        )

        dmte_summary, overall_summary = self.paper_summarizer.summarize(
            paper_content, replace=self.args.replace
        )
        config_yaml_str = self.config_extractor.extract_config(
            paper_content, addendum_section, replace=self.args.replace
        )

        logging.info("--- Stage 2 & 2.5: Generating & Enriching Code Framework ---")
        framework_with_steps = self._generate_framework_with_steps(
            dmte_summary, overall_summary, config_yaml_str, addendum_section
        )
        enriched_framework = self._enrich_framework(framework_with_steps, paper_content)

        logging.info("--- Stage 3: Generating Code Implementation ---")
        final_code, final_script_path = self._generate_implementation(
            enriched_framework, paper_content, addendum_section, config_yaml_str
        )

        logging.info("--- Stage 4: Generating Experiment Code ---")
        self.exp_gen.generate(paper_content, addendum_section, final_code)

        logging.info("=" * 50)
        logging.info("🎉🎉🎉 Code Generation Pipeline COMPLETED SUCCESSFULLY! 🎉🎉🎉")
        logging.info(f"Find the final generated repository in: {self.initial_repo_dir}")
        logging.info("=" * 50)

    def _generate_framework_with_steps(
        self, dmte_summary, overall_summary, config_yaml_str, addendum_section
    ):
        base_framework_path = os.path.join(
            self.intermediates_dir, "code_framework_base.py"
        )
        steps_framework_path = os.path.join(
            self.intermediates_dir, "code_framework_with_steps.py"
        )

        if os.path.exists(steps_framework_path) and not self.args.replace:
            logging.info(
                f"Final framework with steps already exists. Loading from: {steps_framework_path}"
            )
            framework_with_steps = read_file(steps_framework_path)
        else:
            if os.path.exists(base_framework_path) and not self.args.replace:
                logging.info(
                    f"Base framework found. Loading from: {base_framework_path}"
                )
                code_framework = read_file(base_framework_path)
            else:
                logging.info("Base framework not found. Generating a new one...")
                code_framework = self.framework_gen.generate(
                    dmte_summary, overall_summary, addendum_section, base_framework_path
                )

            definitions = extract_definitions_in_order(code_framework)
            sections = parse_dmte_summary(dmte_summary)
            config = yaml.safe_load(config_yaml_str)

            target_parts = {
                "Data": {
                    "found": False,
                    "code": "",
                    "summary": sections["Data"].strip(),
                    "config": config["data"],
                },
                "Model": {
                    "found": False,
                    "code": "",
                    "summary": sections["Model"].strip(),
                    "config": config["model"],
                },
                "Trainer": {
                    "found": False,
                    "code": "",
                    "summary": sections["Training"].strip(),
                    "config": config["training"],
                },
                "Evaluator": {
                    "found": False,
                    "code": "",
                    "summary": sections["Evaluation"].strip(),
                    "config": config["evaluation"],
                },
                "main": {"found": False, "code": ""},
            }

            logging.info("Populating target parts with extracted code definitions...")
            for item in definitions:
                if item["type"] == "class" and item["name"] in target_parts:
                    target_parts[item["name"]]["found"] = True
                    target_parts[item["name"]]["code"] = item["code"]
                elif item["type"] == "function" and item["name"] == "main":
                    target_parts["main"]["found"] = True
                    target_parts["main"]["code"] = item["code"]

            logging.info("Adding step comments to the base framework...")
            conversation_history = []
            for part_name, info in target_parts.items():
                if info["found"]:
                    logging.info(f"Generating steps for section: {part_name}")
                    while True:
                        part_with_steps = self.step_gen.generate(
                            overall_summary=overall_summary,
                            code_framework=code_framework,
                            addendum_section=addendum_section,
                            info=info,
                            conversation_history=conversation_history,
                        )
                        new_definition = extract_definitions_in_order(part_with_steps)
                        if new_definition:
                            new_definition = new_definition[0]
                            for i, item in enumerate(definitions):
                                if item.get("name") == new_definition.get("name"):
                                    definitions[i] = new_definition
                            break
                        else:
                            logging.warning(
                                f"Failed to generate valid steps for {part_name}, retrying..."
                            )

            framework_with_steps = restore_and_save_py_file(
                definitions, steps_framework_path, save=True, add_supplement=False
            )
            logging.info(f"Framework with steps saved to {steps_framework_path}")

        return framework_with_steps

    def _enrich_framework(self, framework_with_steps, paper_content):
        logging.info("--- Stage 2.5: Filling and Enriching Code Framework ---")
        base_framework_path = os.path.join(
            self.initial_repo_dir, "code_framework_base.py"
        )
        ast_save_name = (
            os.path.splitext(os.path.basename(base_framework_path))[0] + "_ast.json"
        )

        enriched_framework = self.framework_processor.process_and_enrich(
            code_framework=framework_with_steps,
            ast_save_name=ast_save_name,
            paper_content=paper_content,
            replace=self.args.replace,
            add_supplement=True,
        )
        logging.info("Framework enrichment complete.")

        return enriched_framework

    def _generate_implementation(
        self, enriched_framework, paper_content, addendum_section, config_yaml_str
    ):
        final_script_path = os.path.join(self.initial_repo_dir, "main.py")

        if os.path.exists(final_script_path) and not self.args.replace:
            logging.info(
                f"Final script already exists. Loading from: {final_script_path}"
            )
            final_code = read_file(final_script_path)
        else:
            logging.info("Preparing data for each code part implementation...")
            definitions = extract_definitions_in_order(enriched_framework)
            import_statements = extract_imports(enriched_framework)
            target_parts = {
                "Data": {"found": False, "code": ""},
                "Model": {"found": False, "code": ""},
                "Trainer": {"found": False, "code": ""},
                "Evaluator": {"found": False, "code": ""},
                "main": {"found": False, "code": ""},
            }
            for item in definitions:
                if item["type"] == "class" and item["name"] in target_parts:
                    target_parts[item["name"]]["found"] = True
                    target_parts[item["name"]]["code"] = item["code"]
                elif item["type"] == "function" and item["name"] == "main":
                    target_parts["main"]["found"] = True
                    target_parts["main"]["code"] = item["code"]

            conversation_history = []
            interaction_log = []
            interaction_log.append("--- Multi-Turn Code Implementation Log ---\n")

            for part_name, info in target_parts.items():
                if info["found"]:
                    logging.info(f"Generating implementation for section: {part_name}")

                    while True:
                        implemented_part_code = self.impl_gen.generate(
                            paper_content=paper_content,
                            code_framework=enriched_framework,
                            addendum_section=addendum_section,
                            config=config_yaml_str,
                            output_path=final_script_path,
                            info=info,
                            import_statements=import_statements,
                            conversation_history=conversation_history,
                            interaction_log=interaction_log,
                            replace=self.args.replace,
                        )

                        new_definitions = extract_definitions_in_order(
                            implemented_part_code
                        )
                        new_import_statements = extract_imports(implemented_part_code)
                        import_statements.extend(new_import_statements)
                        if new_definitions:
                            for new_definition in new_definitions:
                                if "name" in new_definition and new_definition.get(
                                    "type"
                                ) in ["class", "function"]:
                                    for i, existing_definition in enumerate(
                                        definitions
                                    ):
                                        if (
                                            "name" in existing_definition
                                            and existing_definition.get("name")
                                            == new_definition.get("name")
                                            and existing_definition.get("type")
                                            == new_definition.get("type")
                                        ):
                                            definitions[i] = new_definition
                                            break  # Matched and replaced, so stop searching
                                elif new_definition.get("type") == "main":
                                    for i, existing_definition in enumerate(
                                        definitions
                                    ):
                                        if existing_definition.get("type") == "main":
                                            definitions[i] = new_definition
                                            break  # Matched and replaced, so stop searching
                            break
            for definition in definitions:
                if definition.get("type") == "import":
                    definition["code"] = import_statements

            final_code = restore_and_save_py_file(
                definitions, final_script_path, save=True
            )
            logging.info(f"Final implemented script saved to {final_script_path}")

            log_file_path = os.path.join(
                self.intermediates_dir, "interaction_history_implementation.log"
            )
            with open(log_file_path, "w", encoding="utf-8") as f:
                f.writelines(interaction_log)

        return final_code, final_script_path


def main():
    try:
        args = parse_args()
        pipeline = CodeGenerationPipeline(args)
        pipeline.run()
    except Exception as e:
        logging.error(
            f"An unexpected error occurred in the pipeline: {e}", exc_info=True
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
