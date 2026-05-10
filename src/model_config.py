"""
model_config.py
===============
Base model and tokenizer configuration for the Legal Document
Summarization LoRA Fine-Tuning project.

Base Model: mistralai/Mistral-7B-Instruct-v0.3
  - Architecture:    Transformer Decoder-Only (GQA attention)
  - Parameters:      8.03 Billion
  - Context Window:  128,000 tokens
  - Tokenizer:       Tiktoken-based BPE (vocab size: 128,256)
  - License:         Apache 2.0 License

This module is imported by train_lora.py, evaluate.py, and inference.py
to guarantee a single source of truth for all model-related configuration.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import torch
import yaml
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
)

logger = logging.getLogger(__name__)

# ── Project-wide constant — DO NOT change this value in any other file ─────────
BASE_MODEL_ID: str = "mistralai/Mistral-7B-Instruct-v0.3"


@dataclass
class ArchitectureInfo:
    """
    Reference dataclass documenting Mistral-7B-Instruct-v0.3 architecture.
    Used for documentation and sanity-checks only — not loaded from the model.
    """
    model_id: str = BASE_MODEL_ID
    architecture: str = "Transformer Decoder-Only (MistralForCausalLM)"
    num_parameters: str = "8.03 Billion"
    num_layers: int = 32
    hidden_dim: int = 4096
    intermediate_dim: int = 14336
    num_attention_heads: int = 32
    num_kv_heads: int = 8            # Grouped Query Attention (GQA)
    vocab_size: int = 128_256
    context_window: int = 128_000
    rope_theta: float = 500_000.0   # RoPE scaling for long context
    tie_word_embeddings: bool = False
    tokenizer_type: str = "Tiktoken BPE"
    attention_type: str = "Grouped Query Attention (GQA)"
    activation: str = "SiLU (Swish)"
    norm: str = "RMSNorm"
    training_cutoff: str = "December 2023"
    license: str = "Apache 2.0 License"

    # Why chosen for legal summarization:
    selection_rationale: list = field(default_factory=lambda: [
        "128K token context window handles full-length contracts without chunking",
        "GQA reduces KV-cache memory by 75% vs MHA, enabling larger batch sizes",
        "Instruct variant provides pre-aligned instruction-following for structured output",
        "7B size is VRAM-efficient under LoRA (fits on 6GB VRAM with float16)",
        "Strong baseline on legal benchmarks (MMLU-Law: 67.8%)",
        "Open license permits commercial research use",
    ])


class ModelConfig:
    """
    Centralised model and tokenizer loader for all pipeline stages.

    Provides:
      - load_tokenizer(): Returns configured AutoTokenizer
      - load_base_model(): Returns model in float16 for LoRA training
      - load_base_model_fp16(): Returns model in BF16 (evaluation/inference)
      - get_bnb_config(): Returns BitsAndBytesConfig for quantization

    Usage:
        cfg = ModelConfig("configs/training_config.yaml")
        tokenizer = cfg.load_tokenizer()
        model = cfg.load_base_model()
    """

    def __init__(self, config_path: str = "configs/training_config.yaml") -> None:
        with open(config_path) as f:
            full_cfg = yaml.safe_load(f)

        self.model_cfg = full_cfg["model"]
        self.quant_cfg = full_cfg["quantization"]
        self.inference_cfg = full_cfg["inference"]

        # Validate that the config base model matches the project constant
        cfg_model_id = self.model_cfg["base_model_id"]
        if cfg_model_id != BASE_MODEL_ID:
            raise ValueError(
                f"Config base_model_id '{cfg_model_id}' does not match "
                f"project constant BASE_MODEL_ID '{BASE_MODEL_ID}'. "
                f"The same base model must be used throughout the project."
            )

        self.arch_info = ArchitectureInfo()
        logger.info(f"ModelConfig initialised for: {BASE_MODEL_ID}")

    # ── Tokenizer ─────────────────────────────────────────────────────────────

    def load_tokenizer(self) -> PreTrainedTokenizer:
        """
        Load and configure the Mistral-7B tokenizer.

        Key configuration:
          - padding_side = 'right'   → consistent with SFTTrainer requirements
          - pad_token = eos_token    → Mistral has no dedicated pad token
          - add_eos_token = True     → ensures training examples are properly terminated

        Returns:
            Configured AutoTokenizer instance.
        """
        logger.info(f"Loading tokenizer: {BASE_MODEL_ID}")
        tokenizer = AutoTokenizer.from_pretrained(
            BASE_MODEL_ID,
            trust_remote_code=self.model_cfg["trust_remote_code"],
            padding_side="right",   # Required for SFTTrainer compatibility
        )

        # Mistral-7B does not have a dedicated pad token; use EOS as pad
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
            logger.debug("Set pad_token = eos_token for Mistral-7B tokenizer")

        logger.info(
            f"Tokenizer loaded | vocab_size={tokenizer.vocab_size:,} | "
            f"pad_token='{tokenizer.pad_token}'"
        )
        return tokenizer

    # ── Quantization Config ───────────────────────────────────────────────────

    def get_bnb_config(self) -> BitsAndBytesConfig:
        """
        Build the BitsAndBytesConfig (kept for optional quantized inference only).

        Note: quantization is not used during standard LoRA training.
        This config is kept for optional inference-time quantization.

        Config parameters:


        Returns:
            Configured BitsAndBytesConfig instance.
        """
        compute_dtype = (
            torch.float16
            if self.quant_cfg["bnb_4bit_compute_dtype"] == "bfloat16"
            else torch.float16
        )

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=self.quant_cfg["load_in_4bit"],
            bnb_4bit_quant_type=self.quant_cfg["bnb_4bit_quant_type"],       # "nf4"
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=self.quant_cfg["bnb_4bit_use_double_quant"],
        )

        logger.info("BitsAndBytes 4-bit config ready (used only if quantize=True)")
        return bnb_config

    # ── Model Loaders ─────────────────────────────────────────────────────────

    def load_base_model(
        self,
        quantize: bool = True,
    ) -> PreTrainedModel:
        """
        Load Mistral-7B-Instruct-v0.3 in float16 for standard LoRA fine-tuning.

        The model weights are frozen; only LoRA adapter parameters
        (applied separately via train_lora.py) are trainable.

        Args:
            quantize: If True, load with 4-bit quantization (for inference).
                      If False (default for training), load in float16.

        Returns:
            Pre-trained model ready for LoRA adaptation.
        """
        logger.info(
            f"Loading base model: {BASE_MODEL_ID} "
            f"({'4-bit quantized' if quantize else 'float16 LoRA'})"
        )

        kwargs = dict(
            pretrained_model_name_or_path=BASE_MODEL_ID,
            trust_remote_code=self.model_cfg["trust_remote_code"],
            device_map=self.model_cfg["device_map"],
        )

        if quantize:
            kwargs["quantization_config"] = self.get_bnb_config()
        else:
            kwargs["torch_dtype"] = (
                torch.float16
                if self.model_cfg["torch_dtype"] == "bfloat16"
                else torch.float16
            )

        model = AutoModelForCausalLM.from_pretrained(**kwargs)

        # Required for gradient checkpointing compatibility with LoRA
        model.config.use_cache = False
        model.config.pretraining_tp = 1  # Disable tensor parallelism for LoRA

        total_params = sum(p.numel() for p in model.parameters())
        logger.info(
            f"Model loaded | Total parameters: {total_params / 1e9:.2f}B | "
            f"Device map: {self.model_cfg['device_map']}"
        )

        return model

    def print_architecture_info(self) -> None:
        """Print a formatted summary of the base model architecture."""
        info = self.arch_info
        print("\n" + "=" * 60)
        print("BASE MODEL ARCHITECTURE")
        print("=" * 60)
        print(f"  Model ID:           {info.model_id}")
        print(f"  Architecture:       {info.architecture}")
        print(f"  Parameters:         {info.num_parameters}")
        print(f"  Layers:             {info.num_layers}")
        print(f"  Hidden Dim:         {info.hidden_dim:,}")
        print(f"  Attention Heads:    {info.num_attention_heads} (KV: {info.num_kv_heads})")
        print(f"  Vocab Size:         {info.vocab_size:,}")
        print(f"  Context Window:     {info.context_window:,} tokens")
        print(f"  Attention Type:     {info.attention_type}")
        print(f"  Activation:         {info.activation}")
        print(f"  Normalisation:      {info.norm}")
        print(f"  Training Cutoff:    {info.training_cutoff}")
        print(f"  License:            {info.license}")
        print("\nSelection Rationale:")
        for reason in info.selection_rationale:
            print(f"  • {reason}")
        print("=" * 60 + "\n")
