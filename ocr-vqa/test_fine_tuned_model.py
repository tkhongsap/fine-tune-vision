import json
import random
import pandas as pd
import os
from pathlib import Path
from PIL import Image
import urllib.request

SCRIPT_DIR = Path(__file__).resolve().parent

# Load local dataset
print("Loading dataset.json...")
with open(SCRIPT_DIR / 'dataset.json', 'r') as f:
    ds = json.load(f)
print(f"Loaded {len(ds)} images from dataset.json")

# Sample splits
all_keys = list(ds.keys())
random.shuffle(all_keys)

# Sample 150 training, 50 validation, 100 test
num_train = 150
num_val = 50
num_test = 100

train_keys = all_keys[:num_train]
val_keys = all_keys[num_train:num_train+num_val]
test_keys = all_keys[num_train+num_val:num_train+num_val+num_test]

print(f"Sampled {len(train_keys)} training, {len(val_keys)} validation, {len(test_keys)} test examples.")
# (Optional) You can inspect a few samples if you want:
print('Train sample:', train_keys[:3])
print('Val sample:', val_keys[:3])
print('Test sample:', test_keys[:3])

