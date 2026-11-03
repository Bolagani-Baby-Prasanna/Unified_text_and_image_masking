# Benchmarks Summary (Person 1)

## FUNSD (form understanding)
- Source: nielsr/funsd-layoutlmv3 on Hugging Face (original: guillaumejaume.github.io/FUNSD)
- Splits: {'train': 149, 'test': 50}
- Columns: ['id', 'tokens', 'bboxes', 'ner_tags', 'image']
- Labels (7): ['O', 'B-HEADER', 'I-HEADER', 'B-QUESTION', 'I-QUESTION', 'B-ANSWER', 'I-ANSWER']
- Boxes normalised to 0-1000: Yes (observed values range from 7 to 992)

## CORD (receipt understanding)
- Source: nielsr/cord-layoutlmv3 on Hugging Face (original: github.com/clovaai/cord)
- Splits: {'train': 800, 'validation': 100, 'test': 100}
- Columns: ['id', 'words', 'bboxes', 'ner_tags', 'image']
- Number of labels: 61
- Boxes normalised to 0-1000: Yes (observed values range from 0 to 996)

## Notes
- Difference from official split sizes (if any): None. Matches the Hugging Face canonical distribution.
- Problems faced while loading: Native dataset python loader scripts are deprecated and blocked by HF datasets package raising a RuntimeError. Solved by loading pre-converted parquet files from Hugging Face using the revision='refs/convert/parquet' branch.
