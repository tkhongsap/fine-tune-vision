# Bottle Claim Detection Vision Model

## Overview

This project uses vision fine-tuning to create a model that can determine if a bottle is eligible for deposit refund. The model is trained to classify bottle images as either "claimable" or "non-claimable" based on their condition and completeness.

## Project Structure

```
vision-fine-tuning/
├── claim-detection/
│   ├── claim-images/         # Dataset of bottle images
│   │   ├── claimed/          # Images of bottles eligible for refund
│   │   └── unclaimed/        # Images of bottles not eligible for refund
│   ├── data_jsonl/           # Generated JSONL files for training
│   │   ├── train.jsonl       # Training dataset
│   │   ├── val.jsonl         # Validation dataset
│   │   └── test.jsonl        # Test dataset (for local evaluation only)
│   └── utils/                # Utility scripts
├── ocr-vqa/                  # OCR and Visual Question Answering component
└── .env                      # Environment variables (API keys, etc.)
```

## Prerequisites

- Python 3.8+
- OpenAI API key with access to fine-tuning capabilities
- Required packages (see `requirements.txt`)

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd vision-fine-tuning
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   Create a `.env` file in the root directory with the following content:
   ```
   OPENAI_API_KEY=your_openai_api_key
   ```

## Dataset Preparation

The dataset consists of images of bottles classified as:
- **Claimed**: Bottles with complete bodies that can be returned for a deposit refund
- **Unclaimed**: Bottles with incomplete bodies or other defects that cannot be returned

Images are organized in the `claim-detection/claim-images/` directory, with subdirectories for each class.

## Workflow

### 1. Dataset Preparation

The dataset preparation script:
- Loads images from the `claim-images` directory
- Analyzes and displays dataset statistics
- Creates a balanced subset of images
- Splits data into train/validation/test sets
- Converts images to base64 format
- Generates JSONL files in the format required for fine-tuning

### 2. Fine-Tuning

The fine-tuning script:
- Uploads training and validation files to OpenAI
- Monitors file processing status
- Creates and initiates a fine-tuning job
- Provides a job ID for tracking progress

### 3. Model Evaluation

After fine-tuning is complete, the model can be evaluated using the test set to assess performance.

## Usage

### 1. Prepare the Dataset

Run the dataset preparation script:
```bash
python claim-detection/00-prepare-dataset.py
```

This will generate JSONL files in the `claim-detection/data_jsonl/` directory.

### 2. Fine-Tune the Model

Run the fine-tuning script:
```bash
python claim-detection/03-fine-tune.py
```

This will upload your dataset to OpenAI and start the fine-tuning process.

### 3. Monitor Fine-Tuning Progress

Monitor the progress of your fine-tuning job using the OpenAI API:
```python
from openai import OpenAI
import os
import dotenv

dotenv.load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Replace with your job ID
job_id = "ft-..."
status = client.fine_tuning.jobs.retrieve(job_id)
print(status)
```

### 4. Use the Fine-Tuned Model

Once fine-tuning is complete, you can use the model to classify new bottle images:
```python
from openai import OpenAI
import os
import dotenv
import base64

dotenv.load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Replace with your fine-tuned model ID
model_id = "ft:..."

# Load and encode an image
with open("path/to/image.jpg", "rb") as f:
    image_data = base64.b64encode(f.read()).decode("utf-8")

response = client.chat.completions.create(
    model=model_id,
    messages=[
        {
            "role": "system", 
            "content": "You are an assistant that decides whether a bottle can be returned for a deposit refund. Look at the image and answer with exactly one word: 'claimable' or 'non-claimable'."
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Is the bottle claimable?"},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}}
            ]
        }
    ]
)

print(response.choices[0].message.content)
```

## Contributors

- tkhongsap