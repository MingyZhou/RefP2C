import logging
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from sklearn.cluster import AgglomerativeClustering
from tqdm import tqdm

from src.clients.api import get_response
from src.signals.retrieval.base_retriever import EmbeddingRetriever
from src.utils.helper import extract_json_object, load_prompt


class SignalFilter:
    def __init__(
        self,
        workspace_dir: str,
        model: str,
        embedding_model_path: str = "./model/all-MiniLM-L6-v2",
        max_parallel_workers: int = 10,
    ):
        self.workspace_dir = workspace_dir
        self.model = model
        self.embedder = EmbeddingRetriever(model_path=embedding_model_path)
        self.max_workers = max_parallel_workers
        self.banned_criteria_set = {
            "The <fact>AdamW optimizer</fact> is used to train the model <scope>for the dataset Cora</scope>.",
            "A <fact>learning rate of 0.0001</fact> is applied <scope>when using the AdamW optimizer on the Cora dataset</scope>.",
            "A <fact>weight decay of 0.01</fact> is used <scope>when using the AdamW optimizer on the Cora dataset</scope>.",
            "The <fact>Adam optimizer</fact> is used to train the model <scope>for the dataset Citeseer</scope>.",
            "A <fact>learning rate of 0.0002</fact> is applied <scope>when using the Adam optimizer on the Citeseer dataset</scope>.",
            "A <fact>dropout of 0.2</fact> is applied <scope>when using the Adam optimizer on the Citeseer dataset</scope>.",
        }

    def filter(
        self,
        signals_to_process: list,
        paper_content: str,
        distance_threshold: float = 0.5,
    ) -> list:
        if not signals_to_process:
            return []

        logging.info(
            f"Starting filtering process on {len(signals_to_process)} signals..."
        )

        items_after_hardcode_filter, hardcoded_removed = self._apply_hardcoded_filter(
            signals_to_process
        )

        deduplicated_items, clusters = self._deduplicate_by_fact(
            items_after_hardcode_filter, distance_threshold
        )

        final_items, llm_discarded_items = self._filter_by_llm_verdict(
            deduplicated_items, paper_content
        )

        return final_items

    def _apply_hardcoded_filter(self, items: list) -> tuple[list, list]:
        kept_items, removed_items = [], []
        for item in items:
            if item.get("criterion") in self.banned_criteria_set:
                removed_items.append(item)
            else:
                kept_items.append(item)
        logging.info(
            f"Hardcoded filter: Kept {len(kept_items)}, Removed {len(removed_items)} items."
        )
        return kept_items, removed_items

    def _deduplicate_by_fact(
        self, items: list, distance_threshold: float
    ) -> tuple[list, list]:
        if not items:
            return [], []

        def _extract_fact_text(criterion: str) -> str:
            if not criterion:
                return ""
            match = re.search(
                r"<fact>(.*?)</fact>", criterion, re.IGNORECASE | re.DOTALL
            )
            return match.group(1).strip() if match else ""

        texts_to_embed = [
            _extract_fact_text(item.get("criterion", "")) for item in items
        ]
        embeddings = self.embedder._encode_sentences(texts_to_embed)

        logging.info(
            f"Clustering {len(items)} items with threshold: {distance_threshold}..."
        )
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=distance_threshold,
            metric="cosine",
            linkage="average",
        )
        labels = clustering.fit_predict(embeddings)

        clusters = defaultdict(list)
        for i, label in enumerate(labels):
            clusters[label].append(items[i])

        logging.info(
            f"Found {len(clusters)} unique fact clusters. Selecting representatives..."
        )

        final_items = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_cluster = {
                executor.submit(
                    self._select_representative_from_cluster, list(cluster_items)
                ): cluster_items
                for cluster_items in clusters.values()
            }
            for future in tqdm(
                as_completed(future_to_cluster),
                total=len(clusters),
                desc="Deduplicating in Parallel",
            ):
                final_items.extend(future.result())

        return final_items, list(clusters.values())

    def _select_representative_from_cluster(self, cluster_items: list) -> list:
        if not cluster_items:
            return []
        if len(cluster_items) == 1:
            return cluster_items

        system_prompt = load_prompt("filter_select_representative_system")
        items_str = "\n".join(
            [
                f"{i+1}. {item.get('criterion', 'N/A')}"
                for i, item in enumerate(cluster_items)
            ]
        )
        user_prompt = f"From the following list of criteria, select the best representatives:\n\n{items_str}"

        for _ in range(5):  # Retry logic
            try:
                response = get_response(
                    user_prompt, model=self.model, system_prompt_extra=system_prompt
                )
                verdict = extract_json_object(response)
                if verdict and "selected_indices" in verdict:
                    return [
                        cluster_items[i - 1]
                        for i in verdict["selected_indices"]
                        if 1 <= i <= len(cluster_items)
                    ]
            except Exception as e:
                logging.warning(f"Representative selection failed: {e}")

        logging.error(
            f"Failed to select representatives for cluster. Defaulting to first item."
        )
        return [cluster_items[0]]

    def _filter_by_llm_verdict(
        self, items: list, paper_content: str
    ) -> tuple[list, list]:
        analyzer = CriteriaAnalyzer(self.model, paper_content)
        kept_items, discarded_items = [], []

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_item = {
                executor.submit(analyzer.analyze, item.get("criterion")): item
                for item in items
            }
            for future in tqdm(
                as_completed(future_to_item),
                total=len(items),
                desc="Filtering by LLM Verdict",
            ):
                item = future_to_item[future]
                try:
                    verdict = future.result()
                    if verdict.get("verdict") == "keep":
                        kept_items.append(item)
                    else:
                        item["discard_reason"] = verdict.get("reason")
                        item["discard_category"] = verdict.get("category")
                        discarded_items.append(item)
                except Exception as e:
                    logging.error(f"An item failed LLM verdict analysis: {e}")
                    item["discard_reason"] = "Processing Error"
                    item["discard_category"] = "Error"
                    discarded_items.append(item)

        return kept_items, discarded_items


class CriteriaAnalyzer:
    def __init__(self, model: str, full_paper_text: str):
        self.model = model
        self.system_prompt = load_prompt(
            "filter_verdict_system", full_paper_text=full_paper_text
        )

    def analyze(self, criterion: str) -> dict:
        user_prompt = f"Please evaluate this criterion:\n\n`{criterion}`"
        for _ in range(5):  # Retry logic
            try:
                response = get_response(user_prompt, self.model, self.system_prompt)
                verdict = extract_json_object(response)
                if verdict and "verdict" in verdict:
                    return verdict
            except Exception as e:
                logging.warning(f"Verdict analysis failed: {e}")
        return {
            "verdict": "error",
            "reason": "LLM failed to respond.",
            "category": "Processing Error",
        }
