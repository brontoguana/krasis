#!/usr/bin/env python3
"""Quick test: Rust-accelerated CPU decode on QCN."""

import sys
import time
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger("test_rust_decode")

# Enable timing
import os
os.environ["KRASIS_CPU_DECODE_TIMING"] = "1"
os.environ["KRASIS_TIMING_INTERVAL"] = "10"

from krasis.config import ModelConfig, QuantConfig
from krasis.model import KrasisModel

MODEL_PATH = os.path.expanduser("~/.krasis/models/Qwen3-Coder-Next")

def test_qcn():
    qcfg = QuantConfig(
        gpu_expert_bits=4,
        cpu_expert_bits=4,
        attention="bf16",
        lm_head="int8",
        shared_expert="int8",
        dense_mlp="int8",
    )

    logger.info("Loading QCN model...")
    model = KrasisModel(
        model_path=MODEL_PATH,
        num_gpus=1,
        layer_group_size=2,
        quant_cfg=qcfg,
    )
    model.load()
    logger.info("Model loaded, generating...")

    tokenizer = model.tokenizer

    prompt = "What is the capital of France? Answer in one word:"
    prompt_ids = tokenizer.encode(prompt)
    gen_ids = model.generate(prompt_ids, max_new_tokens=30, temperature=0.1)
    tokens = tokenizer.decode(gen_ids)

    logger.info("Generated: %s", tokens)

    # Check for Paris
    if "paris" in tokens.lower():
        logger.info("PASS: Response contains 'Paris'")
    else:
        logger.warning("CHECK: Response does not contain 'Paris': %s", tokens)

    # Second test
    prompt2 = "Write a Python function that returns the sum of two numbers."
    prompt2_ids = tokenizer.encode(prompt2)
    gen2_ids = model.generate(prompt2_ids, max_new_tokens=50, temperature=0.1)
    tokens2 = tokenizer.decode(gen2_ids)
    logger.info("Generated: %s", tokens2)

    if "def" in tokens2:
        logger.info("PASS: Response contains 'def'")
    else:
        logger.warning("CHECK: Response does not contain 'def': %s", tokens2)


if __name__ == "__main__":
    test_qcn()
