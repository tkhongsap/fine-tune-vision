#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Bottle Claim Detection Dataset Preparation for Fine-Tuning

This script processes a dataset of bottle images to prepare it for fine-tuning a vision model
that determines whether a bottle can be returned for a deposit refund.

The script performs the following steps:
1. Loads and analyzes the image dataset
2. Samples a balanced subset of images
3. Splits the data into train/validation/test sets
4. Converts images to base64 format
5. Creates JSONL files in the format required for fine-tuning

Author: Team
Date: 2023
"""

import base64
import json
import os
import random
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from utils.pretty_print import analyze_image_dataset, print_label_distribution


def build_df(img_dir: Path) -> pd.DataFrame:
    """
    Build a DataFrame from image files in the given directory structure.
    
    Args:
        img_dir (Path): Root directory containing claimed/unclaimed subdirectories
        
    Returns:
        pd.DataFrame: DataFrame with image paths and labels
    """
    records = []
    
    # Process both claimed and unclaimed directories
    for label_dir_name in ["claimed", "unclaimed"]:
        label_dir = img_dir / label_dir_name
        if not label_dir.exists():
            print(f"Directory {label_dir} doesn't exist")
            continue
            
        # Recursively find all image files under this label directory
        for root, dirs, files in os.walk(label_dir):
            root_path = Path(root)
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    file_path = root_path / file
                    records.append({
                        "path": str(file_path),
                        "filename": file,
                        "label": label_dir_name
                    })
    
    return pd.DataFrame(records)


def get_mime_type(filename: str) -> str:
    """
    Determine the MIME type of a file based on its extension.
    
    Args:
        filename (str): The filename to check
        
    Returns:
        str: MIME type string
    """
    ext = filename.lower().split('.')[-1]
    if ext in ["jpg", "jpeg"]:
        return "image/jpeg"
    elif ext == "png":
        return "image/png"
    else:
        return "application/octet-stream"


def add_filename_and_base64(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add filename and base64-encoded image data as columns to the DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame with image paths
        
    Returns:
        pd.DataFrame: Updated DataFrame with filename and base64_uri columns
    """
    df = df.copy()
    df["filename"] = df["path"].apply(lambda p: Path(p).name)
    base64_uris = []
    
    for path in tqdm(df["path"], desc="Encoding images to base64"):
        try:
            with open(path, "rb") as f:
                img_bytes = f.read()
            mime = get_mime_type(path)
            b64 = base64.b64encode(img_bytes).decode("utf-8")
            uri = f"data:{mime};base64,{b64}"
            base64_uris.append(uri)
        except Exception as e:
            print(f"Error encoding {path}: {e}")
            base64_uris.append("")
            
    df["base64_uri"] = base64_uris
    return df


def sample_balanced_dataset(df: pd.DataFrame, sample_size: int, seed_value: int) -> pd.DataFrame:
    """
    Sample a balanced subset of the dataset while maintaining class distribution.
    
    Args:
        df (pd.DataFrame): Original DataFrame
        sample_size (int): Target number of samples
        seed_value (int): Random seed for reproducibility
        
    Returns:
        pd.DataFrame: Balanced sample of the dataset
    """
    # Stratified sampling to maintain label distribution
    sampled_df = df.groupby("label", group_keys=False).apply(
        lambda x: x.sample(min(len(x), int(sample_size * len(x) / len(df))), random_state=seed_value)
    )
    
    # If we don't have exactly sample_size due to rounding, adjust
    if len(sampled_df) > sample_size:
        sampled_df = sampled_df.sample(sample_size, random_state=seed_value)
    elif len(sampled_df) < sample_size:
        # This is unlikely but just in case
        remaining = sample_size - len(sampled_df)
        excluded = pd.concat([df, sampled_df]).drop_duplicates(keep=False)
        if len(excluded) >= remaining:
            sampled_df = pd.concat([sampled_df, excluded.sample(remaining, random_state=seed_value)])
    
    return sampled_df


def row_to_chat_json(row: pd.Series, system_prompt: str) -> dict:
    """
    Map one DataFrame row to the 3-turn chat format for fine-tuning.
    
    Args:
        row (pd.Series): Row from the DataFrame
        system_prompt (str): System prompt for the assistant
        
    Returns:
        dict: JSON-formatted conversation for fine-tuning
    """
    return {
        "messages": [
            {
                "role": "system",
                "content": system_prompt.strip()
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Is the bottle claimable?"},
                    {"type": "image_url", "image_url": {"url": row["base64_uri"]}}
                ]
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": row["label"]}]
            }
        ]
    }


def df_to_jsonl(df: pd.DataFrame, out_path: Path, system_prompt: str) -> None:
    """
    Convert DataFrame to JSONL format and save to a file.
    
    Args:
        df (pd.DataFrame): DataFrame to convert
        out_path (Path): Output file path
        system_prompt (str): System prompt for the assistant
    """
    with out_path.open("w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            json_line = row_to_chat_json(row, system_prompt)
            f.write(json.dumps(json_line, ensure_ascii=False) + "\n")
    print(f"Wrote {len(df):>4} lines  →  {out_path}")


def main():
    """Main function to orchestrate the dataset preparation process."""
    # Set up paths
    script_dir = Path.cwd()  # Use current working directory
    img_dir = script_dir / "claim-detection/claim-images"
    
    print(f"Looking for images in: {img_dir}")
    
    # Load and analyze the dataset
    df = build_df(img_dir)
    analyze_image_dataset(df, script_dir, img_dir)
    
    # Configuration parameters
    sample_size = 30  # Limit the dataset size for performance
    seed_value = 142  # Set seed for reproducibility
    
    # Debug print to verify data loading
    print(f"Found {len(df)} images")
    
    if len(df) > 0:
        print("Sample data:")
        print(df.head())
        
        # Print initial label distribution
        print_label_distribution(df)
        
        # Sample a balanced subset if needed
        if len(df) > sample_size:
            df = sample_balanced_dataset(df, sample_size, seed_value)
            print(f"\nLimited to {sample_size} images")
            print_label_distribution(df, "Limited dataset")
        
        # Split into train, validation, and test sets (70/20/10) with stratification
        train_df, tmp_df = train_test_split(df, test_size=0.30, stratify=df["label"], random_state=seed_value)
        val_df, test_df = train_test_split(tmp_df, test_size=1/3, stratify=tmp_df["label"], random_state=seed_value)
        
        # Print split statistics
        print(f"\nData split statistics:")
        print(f"train: {len(train_df)} images ({len(train_df)/len(df)*100:.1f}%)")
        print(f"val: {len(val_df)} images ({len(val_df)/len(df)*100:.1f}%)")
        print(f"test: {len(test_df)} images ({len(test_df)/len(df)*100:.1f}%)")
        
        # Print label distribution for each split
        print_label_distribution(train_df, "Train")
        print_label_distribution(val_df, "Validation")
        print_label_distribution(test_df, "Test")
        
        # Add base64-encoded image data
        train_df = add_filename_and_base64(train_df)
        val_df = add_filename_and_base64(val_df)
        test_df = add_filename_and_base64(test_df)
        
        # Print sample rows to verify encoding
        print("\nSample train row with new columns:")
        print(train_df.iloc[0][["filename", "base64_uri"]])
        print("\nSample val row with new columns:")
        print(val_df.iloc[0][["filename", "base64_uri"]])
        print("\nSample test row with new columns:")
        print(test_df.iloc[0][["filename", "base64_uri"]])
        
        # Define the system prompt for the fine-tuning task
        SYSTEM_PROMPT = """
        You are an assistant that decides whether a bottle can be returned for a deposit refund.
        Look at the image and answer with exactly one word: "claimable" or "non-claimable".
        """
        
        # Create directory for JSONL files
        data_dir = script_dir / "claim-detection/data_jsonl"
        data_dir.mkdir(exist_ok=True)
        
        # Create JSONL files for each split
        df_to_jsonl(train_df, data_dir / "train.jsonl", SYSTEM_PROMPT)
        df_to_jsonl(val_df, data_dir / "val.jsonl", SYSTEM_PROMPT)
        df_to_jsonl(test_df, data_dir / "test.jsonl", SYSTEM_PROMPT)  # Keep only locally
        
    else:
        print("No images found. Check the directory path and structure.")
        print(f"Looking for images in: {img_dir}")
        print(f"Does directory exist: {img_dir.exists()}")
        if img_dir.exists():
            print("Directory contents:")
            for item in img_dir.iterdir():
                print(f"  - {item.name} ({'dir' if item.is_dir() else 'file'})")


if __name__ == "__main__":
    main() 