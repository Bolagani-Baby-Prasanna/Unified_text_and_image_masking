# Pretraining corpus options

## Scope and recommendation

The approved corpus candidate is `docling-project/DocLayNet-v1.1`. The planned
corpus is 200 DocLayNet train pages selected with a fixed seed. Validation and
test pages are excluded from this pretraining corpus. RVL-CDIP is not combined
with DocLayNet in this phase.

DocLayNet v1.1 is preferred because it provides page images, layout-region
boxes, and digital PDF text cells in modern Parquet shards. Its text cells
provide useful text and coordinates for later multimodal preprocessing, while
the dataset is substantially smaller than v1.2 because it does not embed the
original PDF bytes in every record.

## RVL-CDIP versus original DocLayNet

### RVL-CDIP

The `aharley/rvl_cdip` Hugging Face card describes 400,000 grayscale document
images across 16 classes:

- 320,000 train
- 40,000 validation
- 40,000 test

The standard record contains only `image` and `label`. It does not provide
OCR text, word boxes, layout boxes, document IDs, or page numbers. Images are
TIFF and the largest dimension is no more than 1000 pixels. The dataset is
derived from IIT-CDIP and its card links to Legacy Tobacco Document Library
copyright terms.

RVL-CDIP is useful for document-image classification diversity, but it cannot
directly support word-patch alignment or text-aware LayoutLMv3 inputs without
an additional OCR and document-identity pipeline.

### Original DocLayNet

The original `ds4sd/DocLayNet` source provides 80,863 pages from six document
categories:

- 69,375 train
- 6,489 validation
- 4,999 test

Its older dataset script exposes page images, document metadata, and COCO
layout objects. The separate official extra archive contains PDF files and
per-page JSON files with digital PDF text cells.

The original core archive is documented as approximately 28 GiB and the
extra archive as approximately 7.5 GiB. The legacy archive/script workflow is
less suitable for bounded Hugging Face streaming than the v1.1 Parquet
release.

## DocLayNet versions

### v1.1

Hugging Face dataset: `docling-project/DocLayNet-v1.1`

The notebook pins the verified dataset revision
`5e89392376049f4d589ea339ed64468310ed5c3f`.

Record fields:

```text
image
bboxes
category_id
segmentation
area
pdf_cells
metadata
```

The metadata contains:

```text
coco_height, coco_width, collection, doc_category, image_id,
num_pages, original_filename, original_height, original_width,
page_hash, page_no
```

The dataset has 80,863 records in total. The dataset card reports:

- train: 69,375
- val: 6,489
- test: 4,999
- approximately 7.8 GB download size
- approximately 32.7 GB expanded dataset size

The data is stored in 29 train, 3 validation, and 2 test Parquet shards.
Images are directly accessible as page images, normally 1025 x 1025 pixels.

### v1.2

Hugging Face dataset: `docling-project/DocLayNet-v1.2`

It retains the v1.1 fields and adds:

```text
pdf
modalities
```

The `pdf` field embeds the original PDF bytes in every record. The dataset
card reports the same 80,863 page records but approximately 39.8 GB download
size and approximately 41.2 GB expanded dataset size. It uses 72 train, 7
validation, and 6 test shards.

v1.2 is not selected because the embedded PDF is unnecessary for the planned
PNG corpus and increases transfer/storage requirements substantially.

## Dataset-size discrepancies and sources

The official repository, v1.1 card, and v1.2 card all describe 80,863 pages.
The split totals are consistent:

```text
69,375 + 6,489 + 4,999 = 80,863
```

The original repository documentation labels the middle split `validation`,
while the v1.1 Hugging Face Parquet dataset uses `val`. v1.2 uses
`validation` in its Parquet configuration. Code must inspect the actual split
names rather than assume one spelling.

The older Hugging Face page may show a smaller or stale preview count. The
dataset-card split metadata and official repository totals are the authoritative
sources used for planning.

Sources:

- [Original DocLayNet repository](https://github.com/DS4SD/DocLayNet)
- [Original Hugging Face page](https://huggingface.co/datasets/docling-project/DocLayNet)
- [DocLayNet v1.1](https://huggingface.co/datasets/docling-project/DocLayNet-v1.1)
- [DocLayNet v1.2](https://huggingface.co/datasets/docling-project/DocLayNet-v1.2)
- [DocLayNet paper](https://doi.org/10.1145/3534678.3539043)

## Licence

The DocLayNet dataset cards identify the license as
[CDLA-Permissive-1.0](https://cdla.io/permissive-1-0/). The project should
retain the source URL and citation in corpus metadata and should review any
redistribution requirements before publishing images.

RVL-CDIP has different provenance and is linked to Legacy Tobacco Document
Library copyright terms. It should not be treated as having the DocLayNet
license.

## Streaming limitations

DocLayNet v1.1 is stored as remote Parquet shards and supports:

```python
load_dataset(
    "docling-project/DocLayNet-v1.1",
    split="train",
    streaming=True,
)
```

Streaming avoids materializing the complete dataset locally. It does not
guarantee that exactly 200 image bytes are transferred: Parquet readers may
fetch row groups or portions of larger shards. The extraction notebook uses
bounded `shuffle(...).take(...)` calls and never falls back to a
non-streaming load.

If streaming is unavailable, the safe alternatives are:

1. Run the notebook in Google Colab with a current `datasets` package and
   network access.
2. Use a pre-approved, pinned Parquet mirror or official subset.
3. Stop and resolve the access problem before downloading any full archive.

The notebook does not automatically choose a full-download fallback.

## OCR and text-cell differences

DocLayNet layout boxes are COCO-style layout-region boxes in the
1025 x 1025 image coordinate space. They identify regions such as Text,
Table, Picture, Title, and Section-header. They are not word-level boxes.
LayoutLMv3 normalization to 0-1000 is a future preprocessing step; the
corpus metadata preserves the original 1025 x 1025 coordinates.

`pdf_cells` are nested digital PDF text cells associated with layout regions.
A cell can be a heading, line, paragraph fragment, or table value. It may
contain several words or a complete sentence. Image-only/scanned content may
have no digital PDF cell.

Therefore:

- `has_digital_text` records whether PDF cells are present.
- `has_word_level_ocr` is explicitly `False`.
- The corpus must not invent OCR words.
- The notebook preserves layout boxes, category IDs, segmentation, area, and
  nested PDF cells as JSON columns in `page_ids.csv`.
- Later preprocessing may tokenize cell text and assign a cell box to
  subword tokens, but this is a derived representation and must be documented
  as such.
- Text-cell coordinates use the same COCO/page-image coordinate space as the
  1025 x 1025 PNG, so overlays can be checked directly.

This distinction matters to the unified text-image masking objective:
MLM requires a defined tokenization policy, and WPA requires a documented
mapping from text tokens to image patches. The corpus itself does not solve
that mapping.

## Final recommendation

Use `docling-project/DocLayNet-v1.1` for the first 200-page corpus. It has the
best balance of:

- direct page-image access
- layout annotations
- digital text and coordinates
- modern streaming support
- lower transfer/storage cost than v1.2
- stable document/page provenance

Keep v1.2 as a future option if embedded PDFs become necessary for a
PDF-aware preprocessing experiment.

## Questions for the project guide

1. Is the derived policy of assigning a PDF-cell box to tokenized subwords
   acceptable for the first MLM/WPA experiment?
2. Should selected images be committed to Git LFS, stored externally, or
   represented only by reproducible download metadata?
3. Should the corpus include pages without digital PDF text for MIM-only
   examples, or require `has_digital_text=True` for every page?
4. Should a future OCR pass generate word-level boxes, and if so, which OCR
   engine and coordinate normalization policy should be standardized?
