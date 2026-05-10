"""
dataset.py
==========
Dataset loading, preprocessing, and preparation module for the
Legal Document Summarization LoRA Fine-Tuning project.

Pipeline:
    1. Load raw datasets from Hugging Face (Legal Case Reports + LEDGAR)
    2. Compute and print dataset statistics
    3. Clean text (remove artifacts, normalize whitespace)
    4. Filter by quality criteria (length ratios, min/max token counts)
    5. Format examples as instruction-following prompts (Mistral chat template)
    6. Split into train / validation / test sets
    7. Save processed splits to disk for reproducibility

Base Model: mistralai/Mistral-7B-Instruct-v0.3
"""

import logging
import re
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml
from datasets import (
    Dataset,
    DatasetDict,
    concatenate_datasets,
    load_dataset,
    load_from_disk,
)
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)

# ── Mistral system prompt for legal summarization ────────────────────────────
SYSTEM_PROMPT = """You are an expert legal analyst with extensive experience in \
contract law and judicial case analysis. Your task is to produce concise, accurate, \
and professionally structured summaries of legal documents. \
Preserve all critical legal terms, obligations, rights, dates, and party names exactly \
as they appear in the source document. Do not introduce facts not present in the document."""


def build_instruction_prompt(document: str, summary: Optional[str] = None) -> str:
    """
    Format a legal document (and optionally its summary) using the
    Mistral-7B-Instruct chat template for instruction-following fine-tuning.

    Mistral uses [INST] / [/INST] tags instead of Mistral header tokens.

    Args:
        document: The full legal document text (input).
        summary:  The target expert summary (output). None during inference.

    Returns:
        Formatted prompt string ready for tokenization.
    """
    user_content = (
        f"{SYSTEM_PROMPT}\n\n"
        f"Summarize the following legal document into a concise, accurate, "
        f"and professionally structured summary. Preserve all critical legal "
        f"terms, obligations, rights, dates, parties, and key facts.\n\n"
        f"LEGAL DOCUMENT:\n{document}"
    )
    # Mistral instruct format: <s>[INST] user message [/INST] assistant response </s>
    prompt = f"<s>[INST] {user_content} [/INST]"
    if summary is not None:
        prompt += f" {summary}</s>"
    return prompt


class LegalDatasetProcessor:
    """
    End-to-end dataset processor for legal document summarization.

    Loads data from Hugging Face, applies quality filtering and text
    cleaning, formats examples for supervised fine-tuning, and saves
    processed splits to disk.

    Attributes:
        config: Parsed YAML configuration dictionary.
        tokenizer: Mistral tokenizer for token-length measurements.
        raw_dataset: Concatenated raw dataset before processing.
        processed_dataset: DatasetDict with train/val/test splits.
    """

    def __init__(self, config_path: str = "configs/training_config.yaml") -> None:
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        self.ds_cfg = self.config["dataset"]
        self.model_cfg = self.config["model"]
        self.paths = self.config["paths"]

        self.tokenizer: Optional[AutoTokenizer] = None
        self.raw_dataset: Optional[Dataset] = None
        self.processed_dataset: Optional[DatasetDict] = None

        # Load tokenizer for length measurement (no model weights loaded)
        logger.info(f"Loading tokenizer: {self.model_cfg['base_model_id']}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_cfg["base_model_id"],
            trust_remote_code=self.model_cfg["trust_remote_code"],
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    # ── Public Interface ───────────────────────────────────────────────────────

    def load_datasets(self) -> None:
        """
        Load raw datasets from Hugging Face and concatenate into a
        single Dataset object stored in self.raw_dataset.

        Datasets used:
          - joelniklaus/legal_case_document_summarization (case law, multi-jurisdiction)
          - billsum (US Congressional bills with summaries)
        """
        datasets: List[Dataset] = []

        # ── Dataset 1: Legal Case Document Summarization ──────────────────────
        # Real dataset: expert summaries of legal case documents
        logger.info("Loading joelniklaus/legal_case_document_summarization...")
        try:
            case_ds = load_dataset(
                "joelniklaus/legal_case_document_summarization",
                cache_dir=self.ds_cfg["cache_dir"],
                trust_remote_code=True,
            )
            # Concatenate all available splits
            all_splits = [case_ds[s] for s in case_ds.keys()]
            case_combined = concatenate_datasets(all_splits)
            # This dataset has: 'judgment' and 'summary' columns
            col_map = {}
            if "judgment" in case_combined.column_names and "judgment" != "document":
                col_map["judgment"] = "document"
            if col_map:
                case_combined = case_combined.rename_columns(col_map)
            # Keep only document and summary
            keep = [c for c in ["document", "summary"] if c in case_combined.column_names]
            case_combined = case_combined.select_columns(keep)
            if "document" in case_combined.column_names and "summary" in case_combined.column_names:
                datasets.append(case_combined)
                logger.info(f"  legal_case_document_summarization: {len(case_combined):,} examples")
            else:
                logger.warning(f"  Unexpected columns: {case_combined.column_names}")
        except Exception as e:
            logger.warning(f"Could not load legal_case_document_summarization: {e}. Continuing.")

        # ── Dataset 2: BillSum (US Congressional Bills) ───────────────────────
        # Well-known legal summarization benchmark: bill text → expert summary
        logger.info("Loading billsum dataset...")
        try:
            billsum = load_dataset(
                "billsum",
                cache_dir=self.ds_cfg["cache_dir"],
            )
            all_splits = [billsum[s] for s in billsum.keys()]
            bill_combined = concatenate_datasets(all_splits)
            # billsum columns: 'text', 'summary', 'title'
            if "text" in bill_combined.column_names:
                bill_combined = bill_combined.rename_columns({"text": "document"})
            bill_combined = bill_combined.select_columns(["document", "summary"])
            datasets.append(bill_combined)
            logger.info(f"  billsum: {len(bill_combined):,} examples")
        except Exception as e:
            logger.warning(f"Could not load billsum: {e}. Continuing.")

        if not datasets:
            raise RuntimeError(
                "No datasets could be loaded. Check your internet connection."
            )

        self.raw_dataset = concatenate_datasets(datasets)
        logger.info(f"Total raw documents loaded: {len(self.raw_dataset):,}")

    def preprocess(self) -> None:
        """
        Run the full preprocessing pipeline on self.raw_dataset:
          1. Text cleaning
          2. Quality filtering (length ratios, token counts)
          3. Near-duplicate removal (hash-based)
          4. Prompt formatting (Mistral instruction template)
          5. Train / validation / test splitting
        """
        if self.raw_dataset is None:
            raise RuntimeError("Call load_datasets() before preprocess().")

        logger.info("Step 1/5 — Cleaning text...")
        dataset = self.raw_dataset.map(
            self._clean_example,
            batched=False,
            desc="Cleaning text",
        )

        logger.info("Step 2/5 — Filtering by quality criteria...")
        initial_size = len(dataset)
        dataset = dataset.filter(
            self._quality_filter,
            batched=False,
            desc="Quality filtering",
        )
        logger.info(
            f"  Kept {len(dataset):,} / {initial_size:,} examples "
            f"({len(dataset)/initial_size:.1%})"
        )

        logger.info("Step 3/5 — Removing near-duplicates...")
        dataset = self._dedup(dataset)
        logger.info(f"  After dedup: {len(dataset):,} examples")

        logger.info("Step 4/5 — Formatting instruction prompts...")
        dataset = dataset.map(
            self._format_prompt,
            batched=False,
            desc="Formatting prompts",
        )

        logger.info("Step 5/5 — Splitting into train/val/test...")
        self.processed_dataset = self._split(dataset)

    def print_statistics(self) -> None:
        """Print a summary table of dataset statistics to the logger."""
        if self.raw_dataset is None and self.processed_dataset is None:
            logger.warning("No dataset loaded. Call load_datasets() first.")
            return

        dataset = self.processed_dataset or {"all": self.raw_dataset}

        print("\n" + "=" * 60)
        print("DATASET STATISTICS")
        print("=" * 60)

        for split_name, split_data in dataset.items():
            doc_lengths = [
                len(self.tokenizer.encode(ex["document"], add_special_tokens=False))
                for ex in split_data
            ]
            sum_lengths = [
                len(self.tokenizer.encode(ex["summary"], add_special_tokens=False))
                for ex in split_data
            ]
            avg_doc = sum(doc_lengths) / len(doc_lengths)
            avg_sum = sum(sum_lengths) / len(sum_lengths)
            max_doc = max(doc_lengths)
            max_sum = max(sum_lengths)

            print(f"\nSplit: {split_name.upper()}")
            print(f"  Examples:               {len(split_data):>10,}")
            print(f"  Avg document length:    {avg_doc:>10,.0f} tokens")
            print(f"  Max document length:    {max_doc:>10,} tokens")
            print(f"  Avg summary length:     {avg_sum:>10,.0f} tokens")
            print(f"  Max summary length:     {max_sum:>10,} tokens")
            print(f"  Avg compression ratio:  {avg_doc/avg_sum:>10.1f}:1")

        print("=" * 60 + "\n")

    def save_processed_splits(self) -> None:
        """Save processed DatasetDict to disk for reproducible training."""
        if self.processed_dataset is None:
            raise RuntimeError("Call preprocess() before save_processed_splits().")
        out_path = Path(self.paths["processed_data_dir"])
        out_path.mkdir(parents=True, exist_ok=True)
        self.processed_dataset.save_to_disk(str(out_path))
        logger.info(f"Processed dataset saved to {out_path}")

    def load_processed_splits(self) -> DatasetDict:
        """
        Load previously saved processed splits from disk.

        Returns:
            DatasetDict with 'train', 'validation', 'test' keys.
        """
        path = self.paths["processed_data_dir"]
        logger.info(f"Loading processed dataset from {path}")
        self.processed_dataset = load_from_disk(path)
        return self.processed_dataset

    # ── Private Helpers ────────────────────────────────────────────────────────

    def _infer_column_mapping(
        self,
        dataset: Dataset,
        doc_col: str,
        sum_col: str,
    ) -> Dict[str, str]:
        """
        Attempt to map dataset columns to standardised {document, summary} names.
        Tries common aliases if exact names aren't found.
        """
        cols = dataset.column_names
        mapping: Dict[str, str] = {}

        doc_aliases = ["document", "text", "article", "case_text", "content", "input"]
        sum_aliases = ["summary", "catchphrase", "abstract", "highlights", "target", "output"]

        for alias in doc_aliases:
            if alias in cols and alias != doc_col:
                mapping[alias] = doc_col
                break

        for alias in sum_aliases:
            if alias in cols and alias != sum_col:
                mapping[alias] = sum_col
                break

        return mapping

    def _clean_text(self, text: str) -> str:
        """
        Clean raw legal text extracted from PDFs or web sources.

        Steps:
          - Unicode NFC normalization
          - Remove PDF artifacts (page numbers, running headers)
          - Collapse excessive whitespace
          - Strip leading/trailing whitespace
        """
        if not isinstance(text, str):
            return ""

        # 1. Unicode normalisation
        text = unicodedata.normalize("NFC", text)

        # 2. Remove PDF page number artifacts (e.g., "\n12\n", "— Page 5 —")
        text = re.sub(r"\n\s*\d{1,4}\s*\n", "\n", text)
        text = re.sub(r"[-–—]{2,}\s*[Pp]age\s+\d+\s*[-–—]{2,}", "", text)

        # 3. Remove excessive blank lines (keep at most 2 consecutive newlines)
        text = re.sub(r"\n{3,}", "\n\n", text)

        # 4. Remove non-printable control characters (except \n and \t)
        text = re.sub(r"[^\S\n\t ]+", " ", text)

        # 5. Collapse multiple spaces
        text = re.sub(r" {2,}", " ", text)

        # 6. Strip
        return text.strip()

    def _clean_example(self, example: Dict) -> Dict:
        """Apply _clean_text to document and summary fields."""
        return {
            "document": self._clean_text(example.get("document", "")),
            "summary": self._clean_text(example.get("summary", "")),
        }

    def _count_tokens(self, text: str) -> int:
        """Return the number of tokens in text using the project tokenizer."""
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def _quality_filter(self, example: Dict) -> bool:
        """
        Return True if the example meets all quality criteria:
          - Non-empty document and summary
          - Summary token count within [min_summary_tokens, max_summary_tokens]
          - Document:summary length ratio within [min_ratio, max_ratio]
        """
        doc = example.get("document", "")
        summ = example.get("summary", "")

        if len(doc) < 100 or len(summ) < 20:
            return False

        doc_tokens = self._count_tokens(doc)
        sum_tokens = self._count_tokens(summ)

        if sum_tokens < self.ds_cfg["min_summary_tokens"]:
            return False
        if sum_tokens > self.ds_cfg["max_summary_tokens"]:
            return False

        ratio = doc_tokens / max(sum_tokens, 1)
        if ratio < self.ds_cfg["min_ratio"] or ratio > self.ds_cfg["max_ratio"]:
            return False

        return True

    def _dedup(self, dataset: Dataset) -> Dataset:
        """
        Simple hash-based deduplication on the summary field.
        Removes examples whose summary is identical to a previously seen one.
        """
        seen_hashes: set = set()
        keep_indices: List[int] = []

        for idx, example in enumerate(dataset):
            h = hash(example["summary"].strip().lower())
            if h not in seen_hashes:
                seen_hashes.add(h)
                keep_indices.append(idx)

        return dataset.select(keep_indices)

    def _format_prompt(self, example: Dict) -> Dict:
        """
        Format a single example as a Mistral instruction prompt.
        Truncates the document to max_input_tokens if necessary.

        Returns the example with an additional 'prompt' field containing
        the full formatted string (document + target summary).
        """
        # Truncate source document to max_input_tokens
        doc_tokens = self.tokenizer.encode(
            example["document"],
            add_special_tokens=False,
            max_length=self.ds_cfg["max_input_tokens"],
            truncation=True,
        )
        truncated_doc = self.tokenizer.decode(doc_tokens, skip_special_tokens=True)

        prompt = build_instruction_prompt(
            document=truncated_doc,
            summary=example["summary"],
        )

        return {**example, "prompt": prompt, "document_truncated": truncated_doc}

    def _split(self, dataset: Dataset) -> DatasetDict:
        """
        Stratified train / validation / test split.

        Returns:
            DatasetDict with keys: 'train', 'validation', 'test'.
        """
        train_ratio = self.ds_cfg["train_split"]
        val_ratio = self.ds_cfg["val_split"]
        seed = self.ds_cfg["seed"]

        # First split: train vs. (val + test)
        first_split = dataset.train_test_split(
            test_size=round(1 - train_ratio, 6),
            seed=seed,
        )
        train_ds = first_split["train"]

        # Second split: validation vs. test (equal halves of the remainder)
        val_test_split = first_split["test"].train_test_split(
            test_size=0.5,
            seed=seed,
        )

        return DatasetDict(
            {
                "train": train_ds,
                "validation": val_test_split["train"],
                "test": val_test_split["test"],
            }
        )
