from openai import OpenAI
import json
import os
from datasets import load_dataset
import pandas as pd
import base64
import requests
from io import BytesIO
from PIL import Image
import time

# Initialize OpenAI client
client = OpenAI()

# Try to load dataset from Hugging Face
try:
    # Load dataset
    ds = load_dataset("howard-hou/OCR-VQA")
    print(f"Dataset loaded from Hugging Face with splits: {ds.keys()}")
    using_huggingface = True
except Exception as e:
    print(f"Could not load dataset from Hugging Face: {e}")
    using_huggingface = False

# Function to prepare data for fine-tuning from Hugging Face dataset
def prepare_fine_tuning_data_hf(ds, split="train", max_samples=100):
    """
    Prepare data for fine-tuning in OpenAI's format from Hugging Face dataset
    """
    data = []
    
    # Iterate through samples in the dataset
    for i, sample in enumerate(ds[split]):
        if i >= max_samples:
            break
            
        # Extract image, question, and answer
        image_path = sample.get('image_path')
        question = sample.get('question')
        answer = sample.get('answer')
        
        # Skip if any required field is missing
        if not image_path or not question or not answer:
            continue
            
        # For demonstration purposes, we'll use a simple format
        # In a real implementation, you would encode the image
        data.append({
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful assistant that answers questions about images."
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": question},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encode_image(image_path)}"}}
                    ]
                },
                {
                    "role": "assistant", 
                    "content": answer
                }
            ]
        })
    
    return data

# Function to encode image to base64
def encode_image(image_path):
    """
    Encode image to base64 string
    """
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        print(f"Error encoding image {image_path}: {e}")
        return None

# Function to download image from URL
def download_image(url, save_path):
    """
    Download an image from a URL and save it to the specified path
    """
    try:
        response = requests.get(url, stream=True, timeout=10)
        response.raise_for_status()
        
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        return True
    except Exception as e:
        print(f"Error downloading image from {url}: {e}")
        return False

# Load local OCR-VQA dataset from the provided files
def load_local_ocr_vqa():
    """
    Load OCR-VQA dataset from local files
    """
    try:
        with open('dataset.json', 'r') as f:
            dataset = json.load(f)
        
        print(f"Loaded local dataset with {len(dataset.keys())} images")
        return dataset
    except Exception as e:
        print(f"Error loading local dataset: {e}")
        return None

# Prepare fine-tuning data from local OCR-VQA dataset
def prepare_fine_tuning_data_local(dataset, max_samples=100, download_missing=True):
    """
    Prepare data for fine-tuning in OpenAI's format from local dataset
    
    Args:
        dataset: The loaded dataset
        max_samples: Maximum number of samples to prepare
        download_missing: Whether to download missing images
    """
    data = []
    count = 0
    missing_images = 0
    downloaded_images = 0
    
    # Create images directory if it doesn't exist
    os.makedirs('images', exist_ok=True)
    
    # Process each image in the dataset
    for img_id in dataset.keys():
        if count >= max_samples:
            break
        
        # Get image path and URL
        ext = os.path.splitext(dataset[img_id]['imageURL'])[1]
        if not ext:  # If no extension, default to .jpg
            ext = '.jpg'
        
        image_file = f'images/{img_id}{ext}'
        image_url = dataset[img_id]['imageURL']
        
        # Check if image file exists, download if needed
        if not os.path.exists(image_file):
            missing_images += 1
            if download_missing and image_url:
                print(f"Downloading image {img_id} from {image_url}")
                if download_image(image_url, image_file):
                    downloaded_images += 1
                else:
                    # Skip this image if download failed
                    continue
                # Add a small delay to avoid overwhelming the server
                time.sleep(0.2)
            else:
                # Skip this image if we're not downloading missing images
                continue
        
        # Process each question-answer pair for this image
        for q_idx in range(len(dataset[img_id]['questions'])):
            if count >= max_samples:
                break
                
            question = dataset[img_id]['questions'][q_idx]
            answer = dataset[img_id]['answers'][q_idx]
            
            # Skip if image file doesn't exist
            if not os.path.exists(image_file):
                continue
                
            # Create a training example in OpenAI's format
            encoded_image = encode_image(image_file)
            if not encoded_image:
                continue
                
            data.append({
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that answers questions about book covers and documents."
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": question},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}}
                        ]
                    },
                    {
                        "role": "assistant", 
                        "content": answer
                    }
                ]
            })
            
            count += 1
    
    print(f"Found {missing_images} missing images, downloaded {downloaded_images} successfully")
    print(f"Prepared {len(data)} training examples from local dataset")
    return data

# Example of how to create a fine-tuning job
def create_fine_tuning_job(training_data, model="gpt-4-vision-preview"):
    """
    Create a fine-tuning job with OpenAI
    """
    # Save training data to a file
    with open("training_data.jsonl", "w") as f:
        for item in training_data:
            f.write(json.dumps(item) + "\n")
    
    # Upload the file to OpenAI
    try:
        with open("training_data.jsonl", "rb") as f:
            response = client.files.create(
                file=f,
                purpose="fine-tune"
            )
        file_id = response.id
        print(f"File uploaded with ID: {file_id}")
        
        # Create fine-tuning job
        response = client.fine_tuning.jobs.create(
            training_file=file_id,
            model=model,
            suffix="ocr-vqa-assistant"
        )
        print(f"Fine-tuning job created: {response}")
        return response
    except Exception as e:
        print(f"Error creating fine-tuning job: {e}")
        return None

# Main execution
if __name__ == "__main__":
    # Check if the OpenAI API key is set
    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY environment variable is not set. Please set it before running this script.")
        exit(1)
    
    # Determine which dataset to use
    if using_huggingface:
        print("Using Hugging Face dataset for fine-tuning")
        fine_tuning_data = prepare_fine_tuning_data_hf(ds, max_samples=50)
    else:
        print("Using local OCR-VQA dataset for fine-tuning")
        local_dataset = load_local_ocr_vqa()
        if local_dataset:
            # Ask if user wants to download missing images
            download_choice = input("Do you want to download missing images? (yes/no): ")
            download_missing = download_choice.lower() == "yes"
            
            # Set the maximum number of samples
            try:
                max_samples = int(input("Enter the maximum number of samples to use (default: 50): ") or "50")
            except ValueError:
                max_samples = 50
                
            fine_tuning_data = prepare_fine_tuning_data_local(
                local_dataset, 
                max_samples=max_samples, 
                download_missing=download_missing
            )
        else:
            print("Could not load any dataset. Exiting.")
            exit(1)
    
    # Print summary of training data
    if fine_tuning_data:
        print(f"Prepared {len(fine_tuning_data)} examples for fine-tuning")
        
        # Ask if user wants to create a fine-tuning job
        response = input("Do you want to create a fine-tuning job? (yes/no): ")
        if response.lower() == "yes":
            # Choose model for fine-tuning
            model_choice = input("Enter model to fine-tune (default: gpt-4-vision-preview): ") or "gpt-4-vision-preview"
            create_fine_tuning_job(fine_tuning_data, model=model_choice)
        else:
            print("Fine-tuning job creation skipped.")
    else:
        print("No training data was prepared. Please check for errors above.")
    
    print("Fine-tuning preparation script completed.") 