"""SST-2 sentiment classification: CLS / mean / max pooling comparison."""
import argparse
import csv
import gc
import json
import os
from pathlib import Path

import numpy as np
import torch
from datasets import DownloadMode, load_dataset
from transformers import (
    AutoModel,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)

from BERT import DistilBertForSequenceClassification, POOLING_METHODS

ID2LABEL = {0: "NEGATIVE", 1: "POSITIVE"}
LABEL2ID = {value: key for key, value in ID2LABEL.items()}
PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL = PROJECT_DIR / "cache" / "distilbert"
PRETRAINED_MODEL = "distilbert-base-uncased"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train DistilBERT on SST-2 with configurable pooling."
    )
    parser.add_argument("--pooling", choices=[*POOLING_METHODS, "all"], default="cls")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", "--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_length", "--max-length", type=int, default=128)
    parser.add_argument("--model_path", "--model-path", default=str(DEFAULT_MODEL))
    parser.add_argument("--cache_dir", "--cache-dir", default=str(PROJECT_DIR / "cache"))
    parser.add_argument("--output_dir", "--output-dir", default=str(PROJECT_DIR / "ckpt"))
    parser.add_argument("--fp16", action="store_true", help="Enable CUDA mixed precision.")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.lr <= 0:
        parser.error("epochs, batch_size and lr must be positive")
    if not 2 <= args.max_length <= 512:
        parser.error("max_length must be between 2 and 512")
    if args.fp16 and not torch.cuda.is_available():
        parser.error("--fp16 requires CUDA")
    return args


def prepare_model(args):
    """Use local weights; download the base checkpoint on the first run."""
    path = Path(args.model_path)
    if path.resolve() == DEFAULT_MODEL.resolve() and not (path / "config.json").exists():
        print(f"Downloading {PRETRAINED_MODEL} to {path}", flush=True)
        path.mkdir(parents=True, exist_ok=True)
        tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL, cache_dir=args.cache_dir)
        encoder = AutoModel.from_pretrained(PRETRAINED_MODEL, cache_dir=args.cache_dir)
        encoder.save_pretrained(path)
        tokenizer.save_pretrained(path)
        del encoder
    return AutoTokenizer.from_pretrained(args.model_path, cache_dir=args.cache_dir)


def prepare_dataset(args, tokenizer):
    dataset = load_dataset(
        "stanfordnlp/sst2",
        cache_dir=args.cache_dir,
        download_mode=DownloadMode.REUSE_DATASET_IF_EXISTS,
    )

    def preprocess(examples):
        return tokenizer(
            examples["sentence"], truncation=True, max_length=args.max_length
        )

    return {
        split: dataset[split].map(
            preprocess,
            batched=True,
            remove_columns=[c for c in dataset[split].column_names if c != "label"],
        )
        for split in ("train", "validation")
    }


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    if isinstance(predictions, tuple):
        predictions = predictions[0]
    predictions = np.argmax(predictions, axis=1)
    return {"accuracy": float(np.mean(predictions == labels))}


def train_one(args, pooling, tokenizer, tokenized_dataset):
    print(f"\n{'=' * 60}\nPooling: {pooling}\n{'=' * 60}", flush=True)
    # Reset the random seed BEFORE initializing the classifier for each method.
    set_seed(args.seed)
    model = DistilBertForSequenceClassification.from_pretrained(
        args.model_path,
        cache_dir=args.cache_dir,
        num_labels=2,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
    model.config.pooling = pooling

    root = Path(args.output_dir)
    checkpoint_dir = root / f"{pooling}_ckpt"
    model_dir = root / f"{pooling}_model"
    training_args = TrainingArguments(
        output_dir=str(checkpoint_dir),
        overwrite_output_dir=True,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        save_total_limit=1,
        report_to="none",
        seed=args.seed,
        data_seed=args.seed,
        fp16=args.fp16,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )
    train_result = trainer.train()
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_state()
    metrics = trainer.evaluate()
    trainer.save_metrics("eval", metrics)
    trainer.save_model(str(model_dir))
    tokenizer.save_pretrained(model_dir)

    result = {
        "pooling": pooling,
        "accuracy": float(metrics["eval_accuracy"]),
        "eval_loss": float(metrics["eval_loss"]),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "seed": args.seed,
        "max_length": args.max_length,
        "model_path": args.model_path,
        "fp16": args.fp16,
    }
    with (checkpoint_dir / "experiment_result.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"{pooling}: accuracy={result['accuracy']:.6f}, loss={result['eval_loss']:.6f}")
    if pooling == "cls":
        print(f"CLS accuracy > 0.88: {result['accuracy'] > 0.88}")
    print(f"Model saved to: {model_dir}")

    del trainer, model, training_args
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def save_comparison(output_dir):
    """Combine results from either --pooling all or separate invocations."""
    root = Path(output_dir)
    results = []
    for method in POOLING_METHODS:
        path = root / f"{method}_ckpt" / "experiment_result.json"
        if path.is_file():
            with path.open(encoding="utf-8") as f:
                results.append(json.load(f))
    if not results:
        return
    output = root / "pooling_comparison.csv"
    with output.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    print("\nPooling comparison (latest saved results):")
    print(f"{'Pooling':<10} {'Accuracy':>12} {'Eval loss':>12}")
    for row in results:
        print(f"{row['pooling']:<10} {row['accuracy']:>12.6f} {row['eval_loss']:>12.6f}")
    config_keys = ("epochs", "batch_size", "learning_rate", "seed", "max_length", "model_path", "fp16")
    configs = {tuple(row[k] for k in config_keys) for row in results}
    if len(configs) > 1:
        print("Note: saved runs use different settings; compare runs with matching settings.")
    print(f"Comparison saved to: {output}")


def main():
    args = parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    tokenizer = prepare_model(args)
    tokenized_dataset = prepare_dataset(args, tokenizer)
    methods = POOLING_METHODS if args.pooling == "all" else (args.pooling,)
    for method in methods:
        train_one(args, method, tokenizer, tokenized_dataset)
        save_comparison(args.output_dir)


if __name__ == "__main__":
    main()
