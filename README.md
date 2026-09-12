
# Step 1 — Baseline: LayoutLMv3-base fine-tuned on FUNSD

This is the first milestone of the project. Goal: get a working baseline
number before touching the unified-masking pretraining objective ourselves.

## What this does

- Downloads the FUNSD dataset (forms with words, bounding boxes, and entity
  labels already extracted — no OCR step needed for this stage).
- Loads the **off-the-shelf** `microsoft/layoutlmv3-base` checkpoint
  (already pretrained with unified text-image masking by its original
  authors).
- Fine-tunes it for token classification (entity extraction: header,
  question, answer, other).
- Reports precision/recall/F1 — this is our **baseline row** for the
  final results table.

## Setup (on your GPU machine)

```bash
python -m venv venv
source venv/bin/activate        # on Windows: venv\Scripts\activate
pip install -r requirements.txt
```

If you have a CUDA GPU, install the matching PyTorch build first from
https://pytorch.org/get-started/locally/ before `pip install -r requirements.txt`,
so `torch` picks up CUDA support instead of the CPU-only wheel.

## Run

```bash
python train_baseline.py
```

You should see:
```
CUDA available: True
GPU: <your GPU name>
Loading FUNSD dataset...
...

eval_f1: 0.XX
eval_precision: 0.XX
eval_recall: 0.XX
```

Write these numbers down — they go in your report as "Baseline (no
additional pretraining)".

## Notes / things to expect

- First run downloads the FUNSD dataset (~50MB) and the LayoutLMv3-base
  weights (~500MB) from Hugging Face — needs internet access once.
- Default `max_steps=1000` is a quick run (~15-30 min on a single consumer
  GPU). For a stronger baseline to compare against later, raise
  `max_steps` to ~2000-3000 in `train_baseline.py`.
- If you hit a CUDA out-of-memory error, lower
  `per_device_train_batch_size` from 2 to 1.

## What comes after this (Step 2, not in this script yet)

Once this baseline number is recorded, the next milestone is:
1. Collect an unlabeled in-domain document corpus (or use a subset of
   IIT-CDIP).
2. Implement the continued-pretraining loop with the three losses (MLM +
   MIM + word-patch alignment) on top of this same checkpoint.
3. Re-run this exact fine-tuning script on the continued-pretrained
   checkpoint instead of the vanilla one, and compare F1 against this
   baseline.

Let me know once Step 1 gives you numbers and we'll move to Step 2.

# Unified_text_and_image_masking
