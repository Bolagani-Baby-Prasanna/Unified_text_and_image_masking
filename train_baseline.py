"""
Step 1 — Baseline.

Fine-tune the OFF-THE-SHELF microsoft/layoutlmv3-base checkpoint on FUNSD
(form understanding / entity extraction) with NO extra pretraining of our own.

This gives us the baseline F1 score that our later contribution (continued
pretraining with the unified text-image masking + word-patch alignment
objective) needs to beat or match with less data. Run this first and keep
the printed metrics — you'll quote them in your report as the "baseline"
row of your results table.

Usage:
    python train_baseline.py

Requirements (see requirements.txt):
    pip install -r requirements.txt

Expected runtime: ~15-30 min on a single consumer GPU (e.g. RTX 3060/4060)
for the default max_steps=1000. Increase max_steps for a stronger baseline.
"""

import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    LayoutLMv3ForTokenClassification,
    LayoutLMv3Processor,
    TrainingArguments,
    Trainer,
)
from seqeval.metrics import f1_score, precision_score, recall_score

MODEL_NAME = "microsoft/layoutlmv3-base"
OUTPUT_DIR = "./layoutlmv3-funsd-baseline"


def main():
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # ------------------------------------------------------------------
    # 1. Load FUNSD. This preprocessed version already ships OCR words,
    #    normalized bounding boxes, and NER tags aligned per word — so we
    #    don't need to run our own OCR for this first step.
    # ------------------------------------------------------------------
    print("Loading FUNSD dataset...")
    dataset = load_dataset("nielsr/funsd-layoutlmv3")
    label_list = dataset["train"].features["ner_tags"].feature.names
    id2label = {i: l for i, l in enumerate(label_list)}
    label2id = {l: i for i, l in enumerate(label_list)}
    print(f"Labels: {label_list}")

    # ------------------------------------------------------------------
    # 2. Processor bundles the tokenizer + image feature extractor:
    #    it turns (image, words, boxes) into input_ids, bbox, pixel_values
    #    in one call — exactly the multimodal input LayoutLMv3 expects.
    # ------------------------------------------------------------------
    processor = LayoutLMv3Processor.from_pretrained(MODEL_NAME, apply_ocr=False)

    def prepare_examples(examples):
        images = examples["image"]
        words = examples["tokens"]
        boxes = examples["bboxes"]
        word_labels = examples["ner_tags"]
        encoding = processor(
            images,
            words,
            boxes=boxes,
            word_labels=word_labels,
            truncation=True,
            padding="max_length",
        )
        return encoding

    print("Preprocessing train/test splits...")
    train_dataset = dataset["train"].map(
        prepare_examples,
        batched=True,
        remove_columns=dataset["train"].column_names,
    )
    eval_dataset = dataset["test"].map(
        prepare_examples,
        batched=True,
        remove_columns=dataset["test"].column_names,
    )
    train_dataset.set_format("torch")
    eval_dataset.set_format("torch")

    # ------------------------------------------------------------------
    # 3. Load the pretrained backbone with a token-classification head
    #    on top. The backbone weights are the ones already pretrained
    #    with unified text-image masking — we are NOT training that
    #    objective yet, just checking how well it transfers as-is.
    # ------------------------------------------------------------------
    model = LayoutLMv3ForTokenClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(label_list),
        id2label=id2label,
        label2id=label2id,
    )

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=2)

        true_predictions = [
            [id2label[p] for (p, l) in zip(pred, label) if l != -100]
            for pred, label in zip(predictions, labels)
        ]
        true_labels = [
            [id2label[l] for (p, l) in zip(pred, label) if l != -100]
            for pred, label in zip(predictions, labels)
        ]
        return {
            "precision": precision_score(true_labels, true_predictions),
            "recall": recall_score(true_labels, true_predictions),
            "f1": f1_score(true_labels, true_predictions),
        }

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        max_steps=1000,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        learning_rate=1e-5,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        report_to="none",
        fp16=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )

    print("Starting training...")
    trainer.train()

    print("Running final evaluation...")
    metrics = trainer.evaluate()
    print("\n=== BASELINE FUNSD RESULTS (record these) ===")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    trainer.save_model(f"{OUTPUT_DIR}/final")
    print(f"\nModel saved to {OUTPUT_DIR}/final")


if __name__ == "__main__":
    main()
