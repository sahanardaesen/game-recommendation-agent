"""QLoRA fine-tuning script for Qwen2.5-Instruct.

Runs a short, demo-scale LoRA fine-tuning with a 4-bit quantized base model.
Designed for cards with ~8-16 GB VRAM (RTX 4060, T4). On a 4060 the default
settings finish in roughly 15-30 minutes.

Usage:
    # Sanity check the data pipeline on CPU (no model download, no training)
    python remote/finetune.py --dryrun

    # Real training (needs a CUDA GPU)
    python remote/finetune.py --max-steps 100

Run `python remote/finetune.py --help` for all options.
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA = ROOT / "remote" / "data" / "train.jsonl"
DEFAULT_OUTPUT = ROOT / "remote" / "qlora_out"
DEFAULT_MODEL = "Qwen/Qwen2.5-3B-Instruct"

LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]

INSTRUCTION_PREFIX = "User request: "


def load_examples(path: Path) -> list[dict]:
    examples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            examples.append(json.loads(line))
    return examples


def format_example(rec: dict) -> str:
    return (
        f"{INSTRUCTION_PREFIX}{rec['instruction']}\n\n"
        f"Recommended games:\n{rec['response']}\n"
    )


def check_gpu() -> None:
    if not torch.cuda.is_available():
        print(
            "ERROR: no CUDA GPU detected. Use --dryrun on CPU or run this on a "
            "GPU machine (RTX 4060 / T4).",
            file=sys.stderr,
        )
        sys.exit(1)


def build_dataset(examples: list[dict], tokenizer, max_length: int) -> Dataset:
    texts = [format_example(ex) for ex in examples]
    encodings = tokenizer(texts, truncation=True, max_length=max_length, padding=False)

    # Full-sequence loss (instruction included) keeps this demo simple.
    records = [
        {
            "input_ids": ids,
            "attention_mask": mask,
            "labels": list(ids),
        }
        for ids, mask in zip(encodings["input_ids"], encodings["attention_mask"])
    ]
    return Dataset.from_list(records)


def dryrun(data_path: Path, model_name: str, max_length: int) -> None:
    print(f"Loading dataset from {data_path} ...")
    examples = load_examples(data_path)
    if not examples:
        print("ERROR: dataset is empty", file=sys.stderr)
        sys.exit(1)

    print(f"Examples: {len(examples)}")
    print("Sample input ->")
    print(format_example(examples[0]))

    print(f"Loading tokenizer: {model_name} (CPU, no model weights) ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    enc = tokenizer(format_example(examples[0]), truncation=True, max_length=max_length)
    print(f"Tokenizer OK. Token count (first example): {len(enc['input_ids'])}")

    dataset = build_dataset(examples[:2], tokenizer, max_length)
    print(f"Dataset record keys: {sorted(dataset[0].keys())}")
    print(f"input_ids length = {len(dataset[0]['input_ids'])} (labels identical)")
    print("Dry-run OK: dataset + tokenizer pipeline verified. Exiting without training.")


def main() -> None:
    parser = argparse.ArgumentParser(description="QLoRA fine-tune for Qwen2.5")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="JSONL dataset")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Hugging Face model id")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="adapter output dir")
    parser.add_argument("--max-steps", type=int, default=100, help="training steps")
    parser.add_argument("--lr", type=float, default=2e-4, help="learning rate")
    parser.add_argument("--lr-type", "--lora-r", dest="lora_r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--batch-size", "--per-device-batch-size", dest="batch_size", type=int, default=1)
    parser.add_argument("--grad-accum", "--gradient-accumulation-steps", dest="grad_accum", type=int, default=8)
    parser.add_argument("--max-length", "--seq-len", dest="max_length", type=int, default=512)
    parser.add_argument("--dryrun", action="store_true", help="verify data pipeline only, no training")
    args = parser.parse_args()

    if args.dryrun:
        dryrun(args.data, args.model, args.max_length)
        return

    check_gpu()

    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    print(f"Loading base model: {args.model} (4-bit quantization) ...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)
    model.gradient_checkpointing_enable()

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=LORA_TARGET_MODULES,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    examples = load_examples(args.data)
    print(f"Loaded {len(examples)} training examples from {args.data}")
    dataset = build_dataset(examples, tokenizer, args.max_length)

    training_args = TrainingArguments(
        output_dir=str(args.output),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_steps=10,
        lr_scheduler_type="cosine",
        bf16=False,
        fp16=True,
        logging_steps=10,
        save_strategy="no",
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True),
    )

    trainer.train()
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)

    print(f"\nDone. LoRA adapter saved to: {args.output}")
    print(
        "To load it later:\n"
        "    from peft import PeftModel\n"
        f"    model = PeftModel.from_pretrained(base_model, \"{args.output}\")\n"
    )


if __name__ == "__main__":
    main()