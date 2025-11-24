#!/usr/bin/env python3
"""
predict_ced_labels.py

Run DistilBERT CED classifier (CLAIM / EVIDENCE / OTHER) on a large
sentence-level CSV with checkpointing + resume support.

Features:
- Processes sentences in batches
- Saves predictions every N sentences (chunk_size)
- Can resume if interrupted: skips rows already in the output file
- Shows a progress bar with ETA

Input CSV must contain at least:
- sentence_text

Typically you also have:
- company, report_name, year, page, sentence_id

This script will add:
- row_id          (internal stable ID)
- pred_label_id   (0=CLAIM, 1=EVIDENCE, 2=OTHER)
- pred_label      ("CLAIM", "EVIDENCE", "OTHER")
- pred_conf       (max softmax probability)
"""

import argparse
import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm.auto import tqdm

from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
)


class InferenceDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=128):
        """
        df: DataFrame with at least ['row_id', 'sentence_text']
        """
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = str(row["sentence_text"])
        row_id = int(row["row_id"])

        enc = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )

        # squeeze batch dimension
        item = {k: v.squeeze(0) for k, v in enc.items()}
        # keep row_id as a separate field
        item["row_id"] = torch.tensor(row_id, dtype=torch.long)
        return item


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_csv",
        type=str,
        required=True,
        help="Path to CSV with all sentences (e.g., all_corpus_sentences.csv)",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="models/ced/distilbert_ced",
        help="Path to trained DistilBERT CED model directory",
    )
    parser.add_argument(
        "--out_csv",
        type=str,
        required=True,
        help="Output CSV path for predictions (checkpointed/resumable)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for inference",
    )
    parser.add_argument(
        "--max_len",
        type=int,
        default=128,
        help="Max token length for DistilBERT",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=5000,
        help="Number of sentences per write chunk (checkpoint interval)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load input data
    print(f"Loading input from {args.input_csv} ...")
    df = pd.read_csv(args.input_csv)

    if "sentence_text" not in df.columns:
        raise ValueError("Input CSV must contain a 'sentence_text' column.")

    # Add stable row_id (0..N-1) for resume logic
    if "row_id" not in df.columns:
        df = df.reset_index().rename(columns={"index": "row_id"})
    else:
        # ensure it's int
        df["row_id"] = df["row_id"].astype(int)

    total_rows = len(df)
    print(f"Total sentences in input: {total_rows}")

    # 2. Determine which rows are already processed (resume)
    processed_ids = set()
    already_written = False

    if os.path.exists(args.out_csv):
        print(f"Existing output file found at {args.out_csv}, loading row_id to resume...")
        try:
            existing = pd.read_csv(args.out_csv, usecols=["row_id"])
            processed_ids = set(existing["row_id"].astype(int).tolist())
            already_written = True
            print(f"Already processed sentences: {len(processed_ids)}")
        except Exception as e:
            print(f"Warning: Could not read existing output file cleanly: {e}")
            print("Resume may not work as expected; starting fresh.")
            processed_ids = set()
            already_written = False

    # Filter remaining rows
    if processed_ids:
        remaining_df = df[~df["row_id"].isin(processed_ids)].copy()
    else:
        remaining_df = df.copy()

    if remaining_df.empty:
        print("All rows already processed. Nothing to do.")
        return

    print(f"Sentences left to process: {len(remaining_df)}")

    # 3. Load tokenizer + model
    print(f"Loading model and tokenizer from {args.model_dir} ...")
    tokenizer = DistilBertTokenizerFast.from_pretrained(args.model_dir)
    model = DistilBertForSequenceClassification.from_pretrained(args.model_dir)
    model.to(device)
    model.eval()

    # id2label mapping
    id2label = model.config.id2label
    if not id2label:
        id2label = {
            0: "CLAIM",
            1: "EVIDENCE",
            2: "OTHER",
        }
    # normalize keys to int
    norm_id2label = {}
    for k, v in id2label.items():
        try:
            norm_id2label[int(k)] = v
        except Exception:
            continue

    dataset = InferenceDataset(remaining_df, tokenizer, max_len=args.max_len)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    softmax = torch.nn.Softmax(dim=-1)

    buffer = []  # will store dicts with row_id + preds
    processed_in_this_run = 0

    pbar = tqdm(total=len(remaining_df), desc="Predicting", unit="sent")

    def flush_buffer():
        nonlocal already_written, buffer
        if not buffer:
            return
        # turn buffer into df
        preds_df = pd.DataFrame(buffer)

        # merge with original df to get all columns
        merged = preds_df.merge(df, on="row_id", how="left")

        # Desired column order: all input columns + predictions
        input_cols = list(df.columns)
        extra_cols = ["pred_label_id", "pred_label", "pred_conf"]
        cols = input_cols + [c for c in extra_cols if c not in input_cols]
        merged = merged[cols]

        write_mode = "a" if already_written and os.path.exists(args.out_csv) else "w"
        header = not (already_written and os.path.exists(args.out_csv))

        merged.to_csv(args.out_csv, mode=write_mode, header=header, index=False)
        already_written = True
        buffer = []

    # 4. Inference loop with checkpointing
    with torch.no_grad():
        for batch in loader:
            row_ids = batch["row_id"].cpu().numpy()
            batch_inputs = {k: v.to(device) for k, v in batch.items() if k != "row_id"}

            outputs = model(**batch_inputs)
            logits = outputs.logits
            probs = softmax(logits)

            conf, pred_ids = torch.max(probs, dim=-1)

            pred_ids_np = pred_ids.cpu().numpy()
            conf_np = conf.cpu().numpy()

            for rid, pid, c in zip(row_ids, pred_ids_np, conf_np):
                buffer.append(
                    {
                        "row_id": int(rid),
                        "pred_label_id": int(pid),
                        "pred_label": norm_id2label.get(int(pid), str(int(pid))),
                        "pred_conf": float(c),
                    }
                )

            processed_in_this_run += len(row_ids)
            pbar.update(len(row_ids))

            # checkpoint every chunk_size
            if len(buffer) >= args.chunk_size:
                flush_buffer()

        # flush remaining
        flush_buffer()

    pbar.close()
    print(f"Finished. Newly processed sentences in this run: {processed_in_this_run}")
    print(f"Predictions saved to {args.out_csv}")


if __name__ == "__main__":
    main()
