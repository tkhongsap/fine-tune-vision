import base64, cv2, json, random, os
from pathlib import Path
from PIL import Image
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm


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
                    records.append({"path": str(file_path), "label": label_dir_name})
    
    return pd.DataFrame(records)

# Using Path to handle relative paths correctly - going up one directory then to claim-img
script_dir = Path.cwd()  # Use current working directory in notebooks
img_dir = script_dir / "claim-img"
df = build_df(img_dir)

# Debug print to see if we're getting data
print(f"Found {len(df)} images")
if len(df) > 0:
    print("Sample data:")
    print(df.head())
    
    # Print label distribution
    label_counts = df["label"].value_counts()
    print("\nLabel distribution:")
    for label, count in label_counts.items():
        percentage = count/len(df)*100
        print(f"{label}: {count} images ({percentage:.1f}%)")

    # 70 / 20 / 10 split – but ALWAYS stratified by label
    train_df, tmp_df = train_test_split(df, test_size=0.30, stratify=df["label"], random_state=42)
    val_df, test_df  = train_test_split(tmp_df, test_size=1/3, stratify=tmp_df["label"], random_state=42)

    print(f"\nData split statistics:")
    print(f"train: {len(train_df)} images ({len(train_df)/len(df)*100:.1f}%)")
    print(f"val: {len(val_df)} images ({len(val_df)/len(df)*100:.1f}%)")
    print(f"test: {len(test_df)} images ({len(test_df)/len(df)*100:.1f}%)")
    
    # Print label distribution in each split
    print("\nTrain set label distribution:")
    train_label_counts = train_df["label"].value_counts()
    for label, count in train_label_counts.items():
        percentage = count/len(train_df)*100
        print(f"{label}: {count} images ({percentage:.1f}%)")
    
    print("\nValidation set label distribution:")
    val_label_counts = val_df["label"].value_counts()
    for label, count in val_label_counts.items():
        percentage = count/len(val_df)*100
        print(f"{label}: {count} images ({percentage:.1f}%)")
    
    print("\nTest set label distribution:")
    test_label_counts = test_df["label"].value_counts()
    for label, count in test_label_counts.items():
        percentage = count/len(test_df)*100
        print(f"{label}: {count} images ({percentage:.1f}%)")

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
else:
    print("No images found. Check the directory path and structure.")
    print(f"Looking for images in: {img_dir}")
    print(f"Does directory exist: {img_dir.exists()}")
    if img_dir.exists():
        print("Directory contents:")
        for item in img_dir.iterdir():
            print(f"  - {item.name} ({'dir' if item.is_dir() else 'file'})") 