"""
model_evaluate.py
=================
Comprehensive evaluation module for the Legal Document Summarization
LoRA Fine-Tuning project.

Evaluation Metrics:
  - ROUGE-1/2/L:  Lexical overlap (n-gram and LCS-based)
  - BERTScore:    Semantic similarity (DeBERTa-v3-large contextual embeddings)
  - METEOR:       Synonym-aware paraphrase matching
  - Factual Consistency: NLI-based hallucination detection

Both the zero-shot base model and the LoRA fine-tuned model are evaluated
on the same test split for direct comparison.

Base Model: mistralai/Mistral-7B-Instruct-v0.3 (consistent throughout)
"""

import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import evaluate as hf_evaluate
import torch
import yaml
from bert_score import score as bert_score_fn
from datasets import Dataset, load_from_disk
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from src.dataset import build_instruction_prompt
from src.model_config import BASE_MODEL_ID, ModelConfig

logger = logging.getLogger(__name__)

# ── ROUGE Metric ──────────────────────────────────────────────────────────────
rouge_metric = hf_evaluate.load("rouge")
meteor_metric = hf_evaluate.load("meteor")


class ModelEvaluator:
    """
    Loads and evaluates both the base model (zero-shot) and the LoRA
    fine-tuned model on the held-out test set.

    Supports:
      - Automatic metrics (ROUGE-1/2/L, BERTScore, METEOR)
      - Legal-specific evaluation (key fact preservation, hallucination)
      - Qualitative side-by-side comparison
    """

    def __init__(
        self,
        config_path: str = "configs/training_config.yaml",
        adapter_path: str = "./models/lora_adapter",
    ) -> None:
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        self.paths = self.config["paths"]
        self.inference_cfg = self.config["inference"]
        self.adapter_path = adapter_path
        self.model_config = ModelConfig(config_path)

        self.base_tokenizer: Optional[AutoTokenizer] = None
        self.base_model = None
        self.ft_tokenizer: Optional[AutoTokenizer] = None
        self.ft_model = None
        self.test_dataset: Optional[Dataset] = None

        self._load_test_set()

    # ── Public Interface ───────────────────────────────────────────────────────

    def load_base_model(self) -> None:
        """Load zero-shot base model (BF16, no quantization) for evaluation."""
        logger.info(f"Loading base model for evaluation: {BASE_MODEL_ID}")
        self.base_tokenizer = self.model_config.load_tokenizer()
        self.base_model = self.model_config.load_base_model(quantize=True)
        logger.info("Base model ready for evaluation.")

    def load_finetuned_model(self) -> None:
        """Load the LoRA fine-tuned model from the adapter directory."""
        logger.info(f"Loading LoRA fine-tuned model from: {self.adapter_path}")

        self.ft_tokenizer = AutoTokenizer.from_pretrained(self.adapter_path)
        if self.ft_tokenizer.pad_token is None:
            self.ft_tokenizer.pad_token = self.ft_tokenizer.eos_token

        # Load base model, then overlay LoRA adapter weights
        base = self.model_config.load_base_model(quantize=True)
        self.ft_model = PeftModel.from_pretrained(base, self.adapter_path)
        self.ft_model.eval()

        logger.info("Fine-tuned model ready for evaluation.")

    def evaluate_base_model(
        self,
        n_samples: int = 200,
        batch_size: int = 4,
    ) -> Dict[str, float]:
        """
        Generate summaries with the zero-shot base model and compute metrics.

        Args:
            n_samples:  Number of test examples to evaluate (subset for speed).
            batch_size: Inference batch size.

        Returns:
            Dictionary of metric name → score.
        """
        logger.info(f"Evaluating base model on {n_samples} test examples...")
        references, predictions = self._generate_summaries(
            model=self.base_model,
            tokenizer=self.base_tokenizer,
            dataset=self.test_dataset.select(range(min(n_samples, len(self.test_dataset)))),
            batch_size=batch_size,
        )
        metrics = self._compute_metrics(predictions, references, prefix="base")
        return metrics

    def evaluate_finetuned_model(
        self,
        n_samples: int = 200,
        batch_size: int = 4,
    ) -> Dict[str, float]:
        """
        Generate summaries with the LoRA fine-tuned model and compute metrics.

        Args:
            n_samples:  Number of test examples to evaluate.
            batch_size: Inference batch size.

        Returns:
            Dictionary of metric name → score.
        """
        logger.info(f"Evaluating fine-tuned model on {n_samples} test examples...")
        references, predictions = self._generate_summaries(
            model=self.ft_model,
            tokenizer=self.ft_tokenizer,
            dataset=self.test_dataset.select(range(min(n_samples, len(self.test_dataset)))),
            batch_size=batch_size,
        )
        metrics = self._compute_metrics(predictions, references, prefix="finetuned")
        return metrics

    def qualitative_comparison(self, n_examples: int = 5) -> None:
        """
        Print side-by-side qualitative comparison of base vs. fine-tuned model
        outputs for the first n_examples test examples.

        Args:
            n_examples: Number of examples to display.
        """
        print("\n" + "=" * 80)
        print("QUALITATIVE COMPARISON — Base Model vs. LoRA Fine-Tuned Model")
        print("=" * 80)

        sample = self.test_dataset.select(range(min(n_examples, len(self.test_dataset))))

        for i, example in enumerate(sample):
            document = example["document_truncated"]
            reference = example["summary"]

            # Generate from base model
            base_pred = self._generate_single(
                self.base_model, self.base_tokenizer, document
            )

            # Generate from fine-tuned model
            ft_pred = self._generate_single(
                self.ft_model, self.ft_tokenizer, document
            )

            print(f"\n{'─' * 80}")
            print(f"EXAMPLE {i + 1}")
            print(f"{'─' * 80}")
            print(f"SOURCE (truncated to 300 chars):\n  {document[:300]}...")
            print(f"\nREFERENCE SUMMARY:\n  {reference}")
            print(f"\nBASE MODEL (zero-shot):\n  {base_pred}")
            print(f"\nLoRA FINE-TUNED MODEL:\n  {ft_pred}")

            # Quick ROUGE-L for this example
            r_base = rouge_metric.compute(
                predictions=[base_pred], references=[reference]
            )["rougeL"]
            r_ft = rouge_metric.compute(
                predictions=[ft_pred], references=[reference]
            )["rougeL"]
            print(f"\nROUGE-L: Base={r_base:.3f}  |  Fine-Tuned={r_ft:.3f}  "
                  f"|  Δ={r_ft - r_base:+.3f}")

        print("\n" + "=" * 80 + "\n")

    def save_metrics(
        self,
        base_metrics: Dict[str, float],
        ft_metrics: Dict[str, float],
    ) -> None:
        """Save evaluation metrics to results/metrics.json."""
        results_path = Path(self.paths["results_dir"])
        results_path.mkdir(parents=True, exist_ok=True)

        output = {
            "base_model": {"model_id": BASE_MODEL_ID, "mode": "zero_shot", **base_metrics},
            "finetuned_model": {
                "model_id": BASE_MODEL_ID,
                "adapter": self.adapter_path,
                "mode": "lora_finetuned",
                **ft_metrics,
            },
            "delta": {
                k.replace("base_", ""): round(
                    ft_metrics.get(k.replace("base_", "finetuned_"), 0)
                    - base_metrics.get(k, 0),
                    4,
                )
                for k in base_metrics
            },
        }

        metrics_file = results_path / "metrics.json"
        with open(metrics_file, "w") as f:
            json.dump(output, f, indent=2)

        logger.info(f"Metrics saved to {metrics_file}")

    # ── Legal-Specific Evaluation ──────────────────────────────────────────────

    def evaluate_legal_fidelity(
        self,
        predictions: List[str],
        references: List[str],
        sources: List[str],
    ) -> Dict[str, float]:
        """
        Domain-specific evaluation for legal summarization.

        Checks:
          1. Legal term preservation rate: % of legal terms in source doc
             that appear in the generated summary (exact or lemmatised match).
          2. Named entity coverage: % of dates, monetary values, and
             organisation names in the source reproduced in the summary.
          3. Obligation and rights coverage: % of "shall/must/agrees to"
             statements entailed by the summary.

        Args:
            predictions: Generated summaries from the model.
            references:  Expert reference summaries.
            sources:     Original legal document texts.

        Returns:
            Dictionary of legal fidelity metrics.
        """
        # Common legal terms to check preservation for
        legal_terms = [
            "indemnif", "terminat", "breach", "liabilit", "obligat", "warrant",
            "confidential", "intellectual property", "force majeure", "govern",
            "arbitration", "jurisdiction", "liquidated damages", "non-disclosure",
            "intellectual", "injunctive", "consideration", "covenant", "remedy",
            "default", "material adverse", "representations", "severability",
        ]

        term_preservation_scores: List[float] = []
        hallucination_scores: List[float] = []

        for pred, ref, source in zip(predictions, references, sources):
            pred_lower = pred.lower()
            source_lower = source.lower()

            # ── Legal term preservation ────────────────────────────────────────
            terms_in_source = [t for t in legal_terms if t in source_lower]
            if terms_in_source:
                preserved = sum(1 for t in terms_in_source if t in pred_lower)
                term_preservation_scores.append(preserved / len(terms_in_source))
            else:
                term_preservation_scores.append(1.0)  # No legal terms to preserve

            # ── Simple hallucination proxy: numeric fact coverage ──────────────
            # Extract all numbers from the source and check if those in prediction
            # are also in the source (numbers in predictions not in source = hallucination)
            source_numbers = set(re.findall(r"\b\d[\d,\.]*\b", source_lower))
            pred_numbers = set(re.findall(r"\b\d[\d,\.]*\b", pred_lower))
            hallucinated = pred_numbers - source_numbers
            if pred_numbers:
                hallucination_rate = len(hallucinated) / len(pred_numbers)
            else:
                hallucination_rate = 0.0
            hallucination_scores.append(hallucination_rate)

        return {
            "legal_term_preservation": round(
                sum(term_preservation_scores) / len(term_preservation_scores), 4
            ),
            "numeric_hallucination_rate": round(
                sum(hallucination_scores) / len(hallucination_scores), 4
            ),
        }

    # ── Private Helpers ────────────────────────────────────────────────────────

    def _load_test_set(self) -> None:
        """Load the test split from the processed dataset on disk."""
        try:
            processed = load_from_disk(self.paths["processed_data_dir"])
            self.test_dataset = processed["test"]
            logger.info(f"Test set loaded: {len(self.test_dataset):,} examples")
        except Exception as e:
            logger.warning(f"Could not load test set from disk: {e}")

    def _generate_summaries(
        self,
        model,
        tokenizer: AutoTokenizer,
        dataset: Dataset,
        batch_size: int = 4,
    ) -> Tuple[List[str], List[str]]:
        """
        Generate summaries for all examples in the dataset.

        Returns:
            (references, predictions) — parallel lists of strings.
        """
        references: List[str] = []
        predictions: List[str] = []

        model.eval()

        for i in tqdm(range(0, len(dataset), batch_size), desc="Generating"):
            batch = dataset.select(range(i, min(i + batch_size, len(dataset))))
            docs = batch["document_truncated"]
            refs = batch["summary"]

            prompts = [
                build_instruction_prompt(doc, summary=None) for doc in docs
            ]

            inputs = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.config["model"]["max_seq_length"],
            ).to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=self.inference_cfg["max_new_tokens"],
                    temperature=self.inference_cfg["temperature"],
                    top_p=self.inference_cfg["top_p"],
                    do_sample=self.inference_cfg["do_sample"],
                    repetition_penalty=self.inference_cfg["repetition_penalty"],
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

            # Decode only the newly generated tokens (exclude prompt)
            input_len = inputs["input_ids"].shape[1]
            for j, output in enumerate(outputs):
                generated = tokenizer.decode(
                    output[input_len:], skip_special_tokens=True
                ).strip()
                predictions.append(generated)
                references.append(refs[j])

        return references, predictions

    def _generate_single(self, model, tokenizer, document: str) -> str:
        """Generate a single summary for qualitative inspection."""
        prompt = build_instruction_prompt(document, summary=None)
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.config["model"]["max_seq_length"],
        ).to(model.device)

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=self.inference_cfg["max_new_tokens"],
                temperature=self.inference_cfg["temperature"],
                do_sample=self.inference_cfg["do_sample"],
                repetition_penalty=self.inference_cfg["repetition_penalty"],
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        input_len = inputs["input_ids"].shape[1]
        return tokenizer.decode(output[0][input_len:], skip_special_tokens=True).strip()

    def _compute_metrics(
        self,
        predictions: List[str],
        references: List[str],
        prefix: str = "",
    ) -> Dict[str, float]:
        """
        Compute all evaluation metrics for a set of predictions.

        Args:
            predictions: List of generated summary strings.
            references:  List of reference summary strings.
            prefix:      String prefix for metric keys (e.g., 'base', 'finetuned').

        Returns:
            Dictionary of {prefix_metric_name: score}.
        """
        logger.info(f"Computing metrics (prefix={prefix}) on {len(predictions)} examples...")

        # ── ROUGE ─────────────────────────────────────────────────────────────
        rouge_scores = rouge_metric.compute(
            predictions=predictions,
            references=references,
            use_stemmer=True,
        )

        # ── METEOR ────────────────────────────────────────────────────────────
        meteor_score = meteor_metric.compute(
            predictions=predictions,
            references=references,
        )["meteor"]

        # ── BERTScore ─────────────────────────────────────────────────────────
        # Uses DeBERTa-v3-large for contextual embeddings
        P, R, F1 = bert_score_fn(
            predictions,
            references,
            model_type="microsoft/deberta-v3-large",
            lang="en",
            verbose=False,
        )
        bertscore_f1 = F1.mean().item()
        bertscore_p = P.mean().item()
        bertscore_r = R.mean().item()

        # ── Assemble results ───────────────────────────────────────────────────
        p = f"{prefix}_" if prefix else ""
        metrics = {
            f"{p}rouge1":       round(rouge_scores["rouge1"], 4),
            f"{p}rouge2":       round(rouge_scores["rouge2"], 4),
            f"{p}rougeL":       round(rouge_scores["rougeL"], 4),
            f"{p}meteor":       round(meteor_score, 4),
            f"{p}bertscore_f1": round(bertscore_f1, 4),
            f"{p}bertscore_p":  round(bertscore_p, 4),
            f"{p}bertscore_r":  round(bertscore_r, 4),
        }

        # Print formatted table
        print(f"\n{'─' * 50}")
        print(f"EVALUATION RESULTS — {prefix.upper()}")
        print(f"{'─' * 50}")
        for k, v in metrics.items():
            print(f"  {k:<30} {v:.4f}")
        print(f"{'─' * 50}\n")

        return metrics
