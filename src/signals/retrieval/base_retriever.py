import logging

import faiss
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from src.clients.api import get_response
from src.utils.md_processing import MarkdownParser


class EmbeddingRetriever:
    def __init__(self, model_path: str, device: str = "cpu"):
        logging.info(f"Loading sentence embedding model from: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path)
        self.device = torch.device(device)
        self.model.to(self.device)
        self.index = None

    def _encode_sentences(self, sentences: list, batch_size: int = 32) -> np.ndarray:
        all_embeddings = []
        for i in tqdm(range(0, len(sentences), batch_size), desc="Encoding Sentences"):
            batch = sentences[i : i + batch_size]
            encoded_input = self.tokenizer(
                batch, padding=True, truncation=True, return_tensors="pt"
            ).to(self.device)
            with torch.no_grad():
                model_output = self.model(**encoded_input)
            token_embeddings = model_output[0]
            input_mask_expanded = (
                encoded_input["attention_mask"]
                .unsqueeze(-1)
                .expand(token_embeddings.size())
                .float()
            )
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            embedding_batch = sum_embeddings / sum_mask
            normalized_embedding = F.normalize(embedding_batch, p=2, dim=1)
            all_embeddings.append(normalized_embedding.cpu().numpy())
        return np.vstack(all_embeddings)

    def build_index(self, sentences: list):
        if not sentences:
            logging.warning("Cannot build index for an empty list of sentences.")
            return
        logging.info(
            f"Building FAISS index for {len(sentences)} sentences/paragraphs..."
        )
        embeddings = self._encode_sentences(sentences)
        embedding_dim = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(embedding_dim)
        self.index.add(embeddings)
        logging.info(
            f"FAISS index built successfully with {self.index.ntotal} vectors."
        )

    def search(self, query: str, top_k: int) -> tuple:
        if self.index is None:
            logging.error("FAISS index not built. Cannot perform search.")
            return np.array([]), np.array([])
        query_emb = self._encode_sentences([query])
        return self.index.search(query_emb, top_k)


class BaseGuideRetriever:
    def __init__(
        self,
        workspace_dir: str,
        llm_model: str,
        embedding_model_path: str = "./model/all-MiniLM-L6-v2",
    ):
        self.workspace_dir = workspace_dir
        self.llm_model = llm_model
        self.embedding_retriever = EmbeddingRetriever(model_path=embedding_model_path)

    def retrieve_evidence(self, guide_facts: list, paper_content: str) -> list:
        logging.info(f"Starting evidence retrieval for {len(guide_facts)} facts...")

        parser = MarkdownParser(paper_content)
        _, clean_paragraphs, sentences_by_para = parser.get_parsing_results()

        if not clean_paragraphs:
            logging.error(
                "No clean paragraphs extracted from the paper. Retrieval cannot proceed."
            )
            return guide_facts

        self.embedding_retriever.build_index(clean_paragraphs)

        enriched_facts = []
        for fact_item in tqdm(guide_facts, desc="Retrieving Evidence for Facts"):
            query_sentence = fact_item.get("fact_sentence")
            if not query_sentence:
                enriched_facts.append(fact_item)
                continue

            distances, top_para_indices = self.embedding_retriever.search(
                query_sentence, top_k=3
            )

            sentence_candidates = []
            for i, para_idx in enumerate(top_para_indices[0]):
                if para_idx < len(sentences_by_para):
                    sentence_candidates.extend(sentences_by_para[para_idx])

            if not sentence_candidates:
                fact_item["retrieved_evidence"] = []
                enriched_facts.append(fact_item)
                continue

            best_evidence = self._llm_re_rank(query_sentence, sentence_candidates)

            fact_item["retrieved_evidence"] = best_evidence
            enriched_facts.append(fact_item)

        logging.info("Evidence retrieval process complete.")
        return enriched_facts

    def _llm_re_rank(
        self, query_sentence: str, sentence_candidates: list, max_retries: int = 5
    ) -> list:
        if not sentence_candidates:
            return []

        selection_prompt = ""
        for i, s in enumerate(sentence_candidates):
            selection_prompt += f"Sentence {i}: {s}\n"
        selection_prompt += f"\nSummary Fact to Verify: '{query_sentence}'\n"
        selection_prompt += "\nFrom the numbered sentences above, identify the sentence or sentences that are the BEST available match for the 'Summary Fact'.\n"
        selection_prompt += "Your response MUST be a comma-separated list of the corresponding numbers (e.g., '1' or '1, 2, 4'). DO NOT provide any other text."

        for attempt in range(max_retries):
            try:
                response_text = get_response(
                    selection_prompt, model=self.llm_model
                ).strip()
                indices_from_llm = [
                    int(i.strip())
                    for i in response_text.split(",")
                    if i.strip().isdigit()
                ]

                validated_indices = [
                    idx
                    for idx in indices_from_llm
                    if 0 <= idx < len(sentence_candidates)
                ]
                if validated_indices:
                    return [
                        {"sentence": sentence_candidates[idx]}
                        for idx in validated_indices
                    ]
            except (ValueError, Exception) as e:
                logging.warning(
                    f"LLM selection failed on attempt {attempt + 1}: {e}. Retrying."
                )

        logging.error(
            f"LLM selection failed after {max_retries} attempts for query: '{query_sentence[:50]}...'"
        )
        return []
