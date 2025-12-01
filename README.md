# ANLP-ESG

End-to-end pipeline for processing ESG / sustainability reports: download PDFs, extract text and layout, clean pages, build corpora, run topic models (LDA/NMF/BERTopic), and classify sentences for claim/evidence detection.

## Contents
- `data/raw/<company>/`: downloaded ESG PDFs.
- `data/processed/<company>_layout_all/`: extracted pages/blocks/figures/tables plus cleaned text.
- `esg_pdf_layout_extract.py`: layout-aware PDF→CSV extractor (blocks, figures, tables, fallback OCR).
- `esg_clean_pipeline.py`: mines headers/footers, removes noise, writes `corpus_pages_cleaned.csv`.
- `topic_pipeline.py` + `topic_model_step1.py`/`topic_model_step2_bertopic.py`: topic modeling (LDA/NMF/BERTopic) and prevalence summaries.
- `models/ced/predict_ced_labels.py`: run DistilBERT CED classifier (CLAIM/EVIDENCE/OTHER).
- `build_gold_dataset.py`, `compute_kappa.py`: adjudicate annotations and measure agreement.
- `drive_extract.py`: sync PDFs from Google Drive folders into `data/raw`.
- `make_esg_segments_all.py`: concatenate cleaned corpora into a single segment CSV.
- Notebooks under `notebooks/`: cleaning and corpus exploration.

## Setup
1) Python deps (Python ≥3.9):
   ```bash
   pip install -r requirements.txt
   ```
2) System deps for layout extraction:
   - `tesseract`, `poppler`, `ghostscript` (e.g., `brew install tesseract poppler ghostscript` on macOS).
3) Optional: Google Drive auth under `auth/` (`service_account.json` or `credentials.json` + `token.json`).

## Typical Pipeline
1) Download PDFs (optional):
   ```bash
   python drive_extract.py --dest data/raw --company APPLE --folder-id <drive_folder_id>
   ```
2) Extract layout-aware CSVs:
   ```bash
   python esg_pdf_layout_extract.py \
     --pdf-dir data/raw/apple \
     --company apple \
     --out-dir data/processed/apple_layout_all \
     --workers 6 --dpi 200 --min-text-chars 200
   ```
   Outputs: `pages.csv`, `blocks.csv`, `figures.csv`, `table_index.csv` (and per-PDF versions).
3) Clean page text (removes headers/footers/ligatures, optional OCR fallback):
   ```bash
   python esg_clean_pipeline.py --company apple --base data/processed --ocr
   ```
   Output: `data/processed/apple_layout_all/corpus_pages_cleaned.csv`.
4) (Optional) Aggregate segments across companies:
   ```bash
   python make_esg_segments_all.py
   ```
5) Topic modeling:
   - Simple LDA+NMF:
     ```bash
     python topic_model_step1.py \
       --inputs data/processed/apple_layout_all/corpus_pages_cleaned.csv \
       --out-dir reports/topics_apple \
       --n-topics 12
     ```
   - LDA (+ optional BERTopic):
     ```bash
     python topic_pipeline.py \
       --inputs data/processed/apple_layout_all/corpus_pages_cleaned.csv \
       --out-dir reports/topics_apple \
       --n-topics 12 --bertopic
     ```
   - BERTopic-only (compact):
     ```bash
     python topic_model_step2_bertopic.py \
       --inputs data/processed/apple_layout_all/corpus_pages_cleaned.csv \
       --out-dir reports/topics_apple_bt
     ```
   Tuning helpers: `bertopic_tune.py`, `bt_tune_refined.py`; metrics comparison: `compare_lda_bert_metrics.py`, visualizations in `plot_topic_metrics.py`, `compare_lda_bert_visualisation.py`, `topics_compare_step3.py`.
6) CED (Claim/Evidence/Other) workflow:
   - Build gold from dual annotations:
     ```bash
     python build_gold_dataset.py --csv "CED Annotation - Annotation - Combined.csv" --out ced_gold_sentences.csv
     ```
   - Agreement:
     ```bash
     python compute_kappa.py --csv "CED Annotation - Annotation - Combined.csv"
     ```
   - Model inference (checkpoint/resume):
     ```bash
     python models/ced/predict_ced_labels.py \
       --input_csv data/processed/esg_segments_all.csv \
       --out_csv data/processed/esg_segments_all_ced_preds.csv \
       --model_dir models/ced/distilbert_ced \
       --batch_size 32 --chunk_size 5000
     ```



## Notes
- Large data files (PDFs, processed CSVs) are not tracked in git; ensure disk space before running extractors.
- For macOS/Windows multiprocessing safety, scripts already use spawn-safe guards.*** End Patch
