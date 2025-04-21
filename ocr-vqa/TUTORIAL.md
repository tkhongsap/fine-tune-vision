# OpenAI Fine-Tuning Tutorial with OCR-VQA Dataset

This tutorial guides you through fine-tuning an OpenAI vision model using the OCR-VQA dataset, which contains book cover images with questions and answers.

## Prerequisites

- OpenAI API key with access to fine-tuning capabilities
- Python 3.8 or higher
- Required Python packages (install via `pip install -r requirements.txt`)

## Step 1: Set Up Your Environment

First, set your OpenAI API key as an environment variable:

```bash
# Windows
set OPENAI_API_KEY=your_api_key_here

# Linux/Mac
export OPENAI_API_KEY=your_api_key_here
```

## Step 2: Understand the Dataset

The OCR-VQA dataset contains:
- Book cover images
- Questions about the content visible in those images
- Answers to those questions

The dataset is structured to test vision-language models on their ability to understand and extract information from visual text.

## Step 3: Prepare Your Training Data

Run the fine-tuning script:

```bash
python fine_tuning.py
```

The script will:
1. Load the OCR-VQA dataset
2. Prompt you to decide if you want to download missing images
3. Ask you how many samples to use for fine-tuning
4. Format the data according to OpenAI's requirements

## Step 4: Create a Fine-Tuning Job

When prompted, choose whether to create a fine-tuning job. If you select yes:
1. The script will upload your training data to OpenAI
2. Create a fine-tuning job using the specified model (default: gpt-4-vision-preview)
3. Display the job ID and status

## Step 5: Monitor Your Fine-Tuning Job

You can monitor the progress of your fine-tuning job through the OpenAI dashboard or using the OpenAI API:

```python
# To check the status of your fine-tuning job
job_id = "your_job_id"
status = client.fine_tuning.jobs.retrieve(job_id)
print(status)
```

## Step 6: Use Your Fine-Tuned Model

Once the fine-tuning job is complete, you can use your fine-tuned model:

```python
# Example using the fine-tuned model
fine_tuned_model = "your_fine_tuned_model_id"

response = client.chat.completions.create(
    model=fine_tuned_model,
    messages=[
        {"role": "system", "content": "You are a helpful assistant that answers questions about book covers."},
        {"role": "user", "content": [
            {"type": "text", "text": "What is the title of this book?"},
            {"type": "image_url", "image_url": {"url": "path_to_your_image"}}
        ]}
    ]
)

print(response.choices[0].message.content)
```

## Common Issues and Troubleshooting

- **Missing Images**: The script will attempt to download missing images if you choose to do so
- **API Rate Limiting**: If you encounter rate limiting, the script adds small delays between requests
- **Fine-Tuning Costs**: Be aware of OpenAI's pricing for fine-tuning, which can vary based on the model and amount of data

## Next Steps

- Experiment with different system prompts
- Try varying the number of training examples
- Evaluate your fine-tuned model on a test set
- Consider fine-tuning with additional datasets for broader capabilities 