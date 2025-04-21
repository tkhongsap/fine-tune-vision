import json
import random
import pandas as pd
import os
from pathlib import Path
from PIL import Image
from io import BytesIO
import base64
from system_prompt import get_system_prompt

def load_dataset(dataset_path):
    """
    Load the OCR-VQA dataset from JSON file
    
    Args:
        dataset_path: Path to the dataset.json file
        
    Returns:
        Dictionary containing the dataset
    """
    print(f"Loading dataset from {dataset_path}...")
    with open(dataset_path, 'r') as f:
        ds = json.load(f)
    print(f"Loaded {len(ds)} images from dataset")
    return ds

def create_flattened_examples(dataset):
    """
    Create a flattened list of (image_id, question_idx) pairs
    
    Args:
        dataset: Dictionary containing the OCR-VQA dataset
        
    Returns:
        List of tuples (image_id, question_idx)
    """
    flattened_examples = []
    for img_id, img_info in dataset.items():
        questions = img_info.get('questions', [])
        for q_idx in range(len(questions)):
            flattened_examples.append((img_id, q_idx))
    
    return flattened_examples

def sample_examples(flattened_examples, num_train=150, num_val=50, num_test=100, seed=42):
    """
    Sample examples from the flattened list for train, validation, and test sets
    
    Args:
        flattened_examples: List of tuples (image_id, question_idx)
        num_train: Number of training examples
        num_val: Number of validation examples
        num_test: Number of test examples
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_examples, val_examples, test_examples)
    """
    # Shuffle the examples for random sampling
    random.seed(seed)
    random.shuffle(flattened_examples)
    
    # Sample examples for each split
    train_examples = flattened_examples[:num_train]
    val_examples = flattened_examples[num_train:num_train+num_val]
    test_examples = flattened_examples[num_train+num_val:num_train+num_val+num_test]
    
    print(f"Sampled {len(train_examples)} training, {len(val_examples)} validation, {len(test_examples)} test examples.")
    
    return train_examples, val_examples, test_examples

def process_examples(examples, dataset, split_name):
    """
    Process examples for a specific split
    
    Args:
        examples: List of tuples (image_id, question_idx)
        dataset: Dictionary containing the OCR-VQA dataset
        split_name: Name of the split (train, val, test)
        
    Returns:
        List of dictionaries containing processed examples
    """
    print(f"Processing {split_name} data...")
    processed_data = []
    
    for img_id, q_idx in examples:
        img_info = dataset[img_id]
        
        # Get the specific question and answer
        question = img_info['questions'][q_idx]
        answer = img_info['answers'][q_idx]
        
        # Store image ID and URL instead of placeholder bytes
        image_data = {
            'image_id': img_id,
            'image_url': img_info.get('imageURL', 'Unknown URL')
        }
        
        # Add exactly one question-answer pair per example
        processed_data.append({
            'image': image_data,
            'question': question,
            'answer': answer
        })
    
    return processed_data

def convert_to_dataframe(data):
    """
    Convert processed data to pandas DataFrame
    
    Args:
        data: List of dictionaries containing processed examples
        
    Returns:
        Pandas DataFrame
    """
    # Convert to pandas dataframe
    df = pd.DataFrame(data)
    
    # Reset index to create unique IDs
    df = df.reset_index(drop=True)
    
    return df

def display_examples(df, num_examples=3, random_seed=42):
    """
    Display a few examples from the dataset with their questions, answers, and images
    
    Args:
        df: Pandas DataFrame containing the dataset
        num_examples: Number of examples to display
        random_seed: Random seed for reproducibility
    """
    random.seed(random_seed)
    indices = random.sample(range(len(df)), min(num_examples, len(df)))
    
    print(f"\n--- Displaying {len(indices)} random examples ---\n")
    
    for i, idx in enumerate(indices):
        print(f"Example {i+1}:")
        print(f"QUESTION: {df.iloc[idx]['question']}")
        print(f"ANSWER: {df.iloc[idx]['answer']}")
        
        # Print image information
        image_data = df.iloc[idx]['image']
        if image_data is not None:
            print(f"IMAGE ID: {image_data['image_id']}")
            print(f"IMAGE URL: {image_data['image_url']}")
        else:
            print("IMAGE: None")
            
        print("-" * 50)

def encode_image(image, quality=100):
    """
    Encode image to base64 string, ensuring it's in RGB format
    
    Args:
        image: PIL Image object
        quality: JPEG quality (1-100, 100 is highest quality)
        
    Returns:
        Base64 encoded string of the image
    """
    if image.mode != 'RGB':
        image = image.convert('RGB')  # Convert to RGB
    buffered = BytesIO()
    image.save(buffered, format="JPEG", quality=quality) 
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def format_for_vision_fine_tuning(df, images_dir="images", image_quality=95):
    """
    Format the dataset for vision fine-tuning in OpenAI's Chat Completions API format
    
    Args:
        df: Pandas DataFrame containing the dataset with 'image', 'question', and 'answer' columns
        images_dir: Directory containing the images
        image_quality: JPEG quality for encoded images (1-100)
        
    Returns:
        List of dictionaries formatted for OpenAI's Chat Completions API
    """
    formatted_data = []
    
    # Import system prompt from system_prompt.py
    system_prompt = get_system_prompt()
    
    for _, row in df.iterrows():
        image_data = row['image']
        question = row['question']
        answer = row['answer']
        
        # Construct image path
        img_id = image_data['image_id']
        img_url = image_data['image_url']
        ext = os.path.splitext(img_url)[1]
        if not ext:  # If no extension, default to .jpg
            ext = '.jpg'
        
        image_path = f"{images_dir}/{img_id}{ext}"
        
        # Skip if image file doesn't exist
        if not os.path.exists(image_path):
            print(f"Warning: Image {image_path} does not exist, skipping example")
            continue
        
        try:
            # Load image with PIL and encode
            image = Image.open(image_path)
            encoded_image = encode_image(image, quality=image_quality)
            
            # Format in OpenAI's Chat Completions API format
            formatted_example = {
                "messages": [
                    {
                        "role": "system",
                        "content": system_prompt
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
            }
            
            formatted_data.append(formatted_example)
            
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            continue
    
    print(f"Formatted {len(formatted_data)} examples for vision fine-tuning")
    return formatted_data

def main():
    # Set paths
    SCRIPT_DIR = Path(__file__).resolve().parent
    dataset_path = SCRIPT_DIR / 'dataset.json'
    
    # Step 1: Load dataset
    dataset = load_dataset(dataset_path)
    
    # Step 2: Create flattened examples
    flattened_examples = create_flattened_examples(dataset)
    
    # Step 3: Sample examples for train, val, test
    train_examples, val_examples, test_examples = sample_examples(
        flattened_examples, num_train=150, num_val=50, num_test=100
    )
    
    # Step 4: Process examples for each split
    train_data = process_examples(train_examples, dataset, "training")
    val_data = process_examples(val_examples, dataset, "validation")
    test_data = process_examples(test_examples, dataset, "test")
    
    # Step 5: Convert processed data to DataFrames
    print("Converting to pandas DataFrames and processing images...")
    ds_train = convert_to_dataframe(train_data)
    ds_val = convert_to_dataframe(val_data)
    ds_test = convert_to_dataframe(test_data)
    
    # Step 6: Print statistics
    print(f"Processed {len(ds_train)} training examples")
    print(f"Processed {len(ds_val)} validation examples") 
    print(f"Processed {len(ds_test)} test examples")
    
    # Step 7: Verify we have the exact number of examples requested
    assert len(ds_train) == 150, f"Expected 150 training examples, got {len(ds_train)}"
    assert len(ds_val) == 50, f"Expected 50 validation examples, got {len(ds_val)}"
    assert len(ds_test) == 100, f"Expected 100 test examples, got {len(ds_test)}"
    
    # Now you have ds_train, ds_val, ds_test DataFrames ready for further processing

    # Display examples
    display_examples(ds_train, num_examples=3)

    # Format test data for vision fine-tuning
    print("\nFormatting test data for vision fine-tuning...")
    formatted_test_data = format_for_vision_fine_tuning(ds_test, image_quality=95)
    
    # Save formatted test data to a file for easier use with the OpenAI API
    test_data_file = SCRIPT_DIR / 'test_data.jsonl'
    print(f"Saving formatted test data to {test_data_file}")
    with open(test_data_file, 'w') as f:
        for item in formatted_test_data:
            f.write(json.dumps(item) + '\n')
    
    print(f"Test data saved to {test_data_file}")

if __name__ == "__main__":
    main()

