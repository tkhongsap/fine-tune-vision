# OCR-VQA Fine-Tuning for OpenAI Models

This repository contains scripts for fine-tuning OpenAI models on the OCR-VQA dataset, which consists of images from book covers with questions and answers about their visual content.

## Setup

1. Clone this repository
2. Set up your OpenAI API key as an environment variable:
   ```bash
   # For Windows
   set OPENAI_API_KEY=your_api_key_here
   
   # For Linux/Mac
   export OPENAI_API_KEY=your_api_key_here
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Dataset

The code can use either:
1. The OCR-VQA dataset from Hugging Face (`howard-hou/OCR-VQA`)
2. The local OCR-VQA dataset in this folder

The local dataset structure contains:
- `dataset.json`: Contains image URLs, questions, and answers
- `images/`: Folder that should contain downloaded images (you may need to create this folder)

## Running the Fine-Tuning

To run the fine-tuning process, navigate to the ocr-vqa directory and run:

```bash
cd ocr-vqa
python fine_tuning.py
```

The script will:
1. Attempt to load the dataset (from Hugging Face or locally)
2. Prepare the training data in OpenAI's required format
3. Ask if you want to create a fine-tuning job
4. If yes, upload the data to OpenAI and start a fine-tuning job

## Notes

- Fine-tuning with vision capabilities requires using models that support vision inputs (like GPT-4 Vision)
- Make sure you have sufficient credits in your OpenAI account for fine-tuning
- The script defaults to using a maximum of 50 samples for fine-tuning, which you can adjust in the code

## File Structure

- `fine_tuning.py`: The main script for fine-tuning
- `dataset.json`: The dataset file containing image URLs, questions, and answers
- `loadDataset.py`: Original script for loading and exploring the dataset
- `images/`: Folder where images should be stored 