# LayoutLMv3 – Paper Notes

Paper: Yupan Huang, Tengchao Lv, Lei Cui, Yutong Lu, and Furu Wei. LayoutLMv3: Pre-training for Document AI with Unified Text and Image Masking (2022), arXiv:2204.08387; ACM Multimedia 2022.
Paper link: https://arxiv.org/abs/2204.08387
PDF: https://arxiv.org/pdf/2204.08387

## 1. What problem does the paper solve, and what was wrong with earlier LayoutLM versions?
   Document AI systems need to understand more than plain text. They must use the words on a page, where those words appear (their 2D layout), and the page's visual appearance. This is important for scanned forms, receipts, invoices, reports, and document question answering.

 Earlier versions had limitations:
 - LayoutLM mainly combined text and 2D layout information. Visual features were not integrated in the same unified way as in later versions.
 - LayoutLMv2 added image information, but relied on CNN grid features (ResNeXt101-FPN). Earlier approaches could also use object detectors such as Faster R-CNN. These visual pipelines add computation, parameters, and/or region-supervision requirements.
 - Multimodal models often trained text and images with different objectives. Text was trained by predicting masked words, while image objectives varied (for example, reconstructing pixels or regressing region features). This mismatch makes it harder to learn aligned text-image representations.
  LayoutLMv3's solution: use one Transformer for text and image tokens, represent images as simple linearly projected patches rather than CNN/object-detection features, and use unified discrete-token objectives (MLM and MIM) plus Word-Patch Alignment (WPA).

## 2. How are text tokens, 2D layout and image patches combined into one input sequence?
   The model receives a document image and its OCR output.
 - Text tokens: OCR extracts the words and their bounding boxes. The text is tokenized (BPE in the experiments). Word embeddings are initialized from RoBERTa.
 - 2D layout: each text token receives position information. The model combines 1D sequence positions with 2D layout features derived from the bounding box: x-coordinate, y-coordinate, width, and height. LayoutLMv3 uses segment-level layout positions, so words in the same segment can share the same 2D position.
 - Image patches: resize the page image to 224 × 224, split it into 16 × 16 patches, flatten the patches, and linearly project each patch to the Transformer hidden size. This gives 14 × 14 = 196 visual patch tokens. Learnable 1D position embeddings are added to the patch embeddings.
 - Fusion: concatenate the text-token sequence and image-patch sequence and feed the combined sequence into the same multimodal Transformer. Self-attention lets text tokens and image tokens exchange information. The model also uses relative 1D and spatial 2D position biases in attention.
 In short: OCR words + their layout embeddings + visual patch embeddings → one shared Transformer sequence.

## 3. MLM: what is masked and how?
   Masked Language Modeling (MLM) masks about 30% of text tokens. It uses span masking: groups of consecutive tokens are selected, with span lengths sampled from a Poisson distribution (λ = 3). The model predicts the original vocabulary token for each masked text position using the surrounding text and image context. The layout information is kept unchanged, helping the model connect words with their positions on the page.

## 4. MIM: what is masked, and what does the model predict?
   Masked Image Modeling (MIM) masks about 40% of image patch tokens, using blockwise masking. The model predicts the original discrete visual token/code for each masked patch. These target codes come from a pre-trained image tokenizer (the paper initializes it from DiT; the visual vocabulary has 8,192 codes). It does not simply reconstruct raw pixels. Predicting discrete visual codes encourages the model to learn meaningful visual and layout structure.

## 5. WPA (word-patch alignment): what is the label, and why does it help?
   Each OCR word has a corresponding location in the document image and therefore one or more corresponding image patches. WPA is a binary classification task for each eligible (unmasked) text token:
 - Aligned (positive): the word is unmasked and its corresponding image patch token(s) are also unmasked.
 - Unaligned (negative): the word is unmasked but its corresponding image patch token(s) are masked.
 Masked text tokens are excluded from the WPA loss. A two-layer MLP predicts the aligned/unaligned label using contextual text and image representations.

**Why it helps:** MLM teaches the model about words; MIM teaches it about visual patches. WPA explicitly teaches whether a word and the patch at its page location correspond. This strengthens fine-grained cross-modal grounding—for example, linking a printed amount to the correct region of a receipt.

## 6. What are the main results, and on which datasets?
   1. **FUNSD:** Form understanding and entity labeling.
   - Large model F1 score: 92.08

2. **CORD:** Receipt understanding and key information extraction.
   - Large model F1 score: 97.46

3. **RVL-CDIP:** Document image classification.
   - Large model accuracy: 95.93%

4. **DocVQA:** Document visual question answering.
   - Large model ANLS score: 83.37

5. **PubLayNet:** Document layout analysis.
   - Base model overall mAP: 95.1

  i) The main comparison table reports Base results of 90.29 F1 (FUNSD), 96.56 F1 (CORD), 95.44% accuracy (RVL-CDIP), and 78.76 ANLS (DocVQA). The Large model reports 92.08, 97.46, 95.93%, and 83.37 respectively. On PubLayNet, the Base model reaches 95.1 overall mAP in the vision-task experiment.
  ii) The paper also reports that LayoutLMv3 is more parameter-efficient than LayoutLMv2 in the Base comparison: 133M vs. 200M parameters. The ablation study indicates that adding MIM and WPA improves performance over using MLM alone, and MIM is especially important for stable training on the visual layout-analysis task.
  **Overall takeaway:** LayoutLMv3 performs strongly on both text-centric tasks (forms, receipts, document QA) and image-centric tasks (document classification and layout analysis), while using a simpler patch-based visual input pipeline.

 ## 7. Three doubts for guide meeting:
 - OCR dependency: Since LayoutLMv3 uses OCR text and bounding boxes, how much would recognition errors or inaccurate bounding boxes affect the model's final predictions?
 - WPA mapping: When a word covers more than one image patch, how exactly are the corresponding patches combined to decide the aligned/unaligned label?
 - Practical trade-off: The paper reports strong results with fewer parameters than LayoutLMv2. What are the main real-world trade-offs in inference speed, memory use, and accuracy when choosing Base versus Large?

References
- Huang et al. (2022), LayoutLMv3: Pre-training for Document AI with Unified Text and Image Masking, arXiv:2204.08387. https://arxiv.org/abs/2204.08387
- Official Hugging Face model documentation: https://huggingface.co/docs/transformers/model_doc/layoutlmv3
