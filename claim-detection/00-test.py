import base64, json, random, os
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from utils.pretty_print import analyze_image_dataset, print_label_distribution

def build_df(img_dir: Path):
    records = []
    
    # We now know the folder structure is more complex with nested folders
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
                    # Store just the filename and the label
                    records.append({
                        "path": str(file_path),
                        "filename": file,
                        "label": label_dir_name
                    })
    
    return pd.DataFrame(records)

# Using Path to handle relative paths correctly
script_dir = Path.cwd()  # Use current working directory in notebooks

# Set path to the claim-images directory
img_dir = script_dir / "claim-detection/claim-images"  # Fix the path to claim-images folder

print(f"Looking for images in: {img_dir}")
df = build_df(img_dir)

# Analyze the dataset and print results
analyze_image_dataset(df, script_dir, img_dir)


# Limit the dataset to x images for perfomrance and add seed value for reproducibility
sample_size = 30    
seed_value = 142 

# Debug print to see if we're getting data
print(f"Found {len(df)} images")
if len(df) > 0:
    print("Sample data:")
    print(df.head())
    
    # Print label distribution using the utility function
    print_label_distribution(df)

    # Limit to only sample_size images, but maintain class balance
    if len(df) > sample_size:
        # Stratified sampling to maintain label distribution
        df = df.groupby("label", group_keys=False).apply(
            lambda x: x.sample(min(len(x), int(sample_size * len(x) / len(df))), random_state=seed_value)
        )
        # If we don't have exactly sample_size due to rounding, adjust
        if len(df) > sample_size:
            df = df.sample(sample_size, random_state=seed_value)
        elif len(df) < sample_size:
            # This is unlikely but just in case
            remaining = sample_size - len(df)
            excluded = pd.concat([df, df]).drop_duplicates(keep=False)
            if len(excluded) >= remaining:
                df = pd.concat([df, excluded.sample(remaining, random_state=seed_value)])
        
        print(f"\nLimited to {sample_size} images")
        
        # Print updated label distribution
        print_label_distribution(df, "Limited dataset")

    # 70 / 20 / 10 split – but ALWAYS stratified by label
    train_df, tmp_df = train_test_split(df, test_size=0.30, stratify=df["label"], random_state=seed_value)
    val_df, test_df  = train_test_split(tmp_df, test_size=1/3, stratify=tmp_df["label"], random_state=seed_value)

    print(f"\nData split statistics:")
    print(f"train: {len(train_df)} images ({len(train_df)/len(df)*100:.1f}%)")
    print(f"val: {len(val_df)} images ({len(val_df)/len(df)*100:.1f}%)")
    print(f"test: {len(test_df)} images ({len(test_df)/len(df)*100:.1f}%)")
    
    # Print label distribution for each split using the utility function
    print_label_distribution(train_df, "Train")
    print_label_distribution(val_df, "Validation")
    print_label_distribution(test_df, "Test")

    # --- Add filename and base64_uri columns ---
    def get_mime_type(filename):
        ext = filename.lower().split('.')[-1]
        if ext in ["jpg", "jpeg"]:
            return "image/jpeg"
        elif ext == "png":
            return "image/png"
        else:
            return "application/octet-stream"

    def add_filename_and_base64(df):
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

    train_df = add_filename_and_base64(train_df)
    val_df = add_filename_and_base64(val_df)
    test_df = add_filename_and_base64(test_df)

    print("\nSample train row with new columns:")
    print(train_df.iloc[0][["filename", "base64_uri"]])
    print("\nSample val row with new columns:")
    print(val_df.iloc[0][["filename", "base64_uri"]])
    print("\nSample test row with new columns:")
    print(test_df.iloc[0][["filename", "base64_uri"]])
    
    # --- Prepare JSONL datasets for fine-tuning ---
    SYSTEM_PROMPT = """
    You are an assistant that decides whether a bottle can be returned for a deposit refund.
    Look at the image and answer with exactly one word: "claimable" or "non-claimable".
    """

    def row_to_chat_json(row: pd.Series) -> dict:
        """
        Map one DataFrame row to the 3-turn chat format:
          system  - your fixed instructions
          user    - ALWAYS the same question + the image
          assistant - ground-truth (claimable / non-claimable)
        """
        return {
            "messages": [
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT.strip()
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

    def df_to_jsonl(df: pd.DataFrame, out_path: Path):
        with out_path.open("w", encoding="utf-8") as f:
            for _, row in df.iterrows():
                json_line = row_to_chat_json(row)
                f.write(json.dumps(json_line, ensure_ascii=False) + "\n")
        print(f"Wrote {len(df):>4} lines  →  {out_path}")

    # Create directory for JSONL data files
    data_dir = script_dir / "claim-detection/data_jsonl"
    data_dir.mkdir(exist_ok=True)

    # Create JSONL files for each split
    df_to_jsonl(train_df, data_dir / "train.jsonl")
    df_to_jsonl(val_df, data_dir / "val.jsonl")
    df_to_jsonl(test_df, data_dir / "test.jsonl")  # keep only locally
    
else:
    print("No images found. Check the directory path and structure.")
    print(f"Looking for images in: {img_dir}")
    print(f"Does directory exist: {img_dir.exists()}")
    if img_dir.exists():
        print("Directory contents:")
        for item in img_dir.iterdir():
            print(f"  - {item.name} ({'dir' if item.is_dir() else 'file'})") 