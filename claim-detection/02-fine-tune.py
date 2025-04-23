#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
OpenAI Vision Model Fine-Tuning Script

This script handles the fine-tuning process for a vision model that determines
whether bottles are claimable for deposit refunds. It performs the following steps:
1. Loads environment variables and API key
2. Uploads training and validation JSONL files to OpenAI
3. Monitors file processing status
4. Creates and initiates a fine-tuning job

Author: Team
Date: 2023
"""

import json
import os
import time
from pathlib import Path

import dotenv
from openai import OpenAI


def load_api_key():
    """
    Load OpenAI API key from environment variables.
    
    Returns:
        str: OpenAI API key
    """
    dotenv.load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables")
        
    return api_key


def upload_file(client, file_path, purpose="fine-tune"):
    """
    Upload a file to OpenAI API.
    
    Args:
        client: OpenAI client instance
        file_path (str): Path to the file to upload
        purpose (str): Purpose of the file (default: "fine-tune")
        
    Returns:
        OpenAI file object
    """
    print(f"Uploading {file_path}...")
    try:
        with open(file_path, "rb") as file:
            return client.files.create(file=file, purpose=purpose)
    except Exception as e:
        print(f"Error uploading file: {e}")
        raise


def wait_for_file_processing(client, file_id, description="file"):
    """
    Wait for a file to be processed by OpenAI.
    
    Args:
        client: OpenAI client instance
        file_id (str): ID of the file to monitor
        description (str): Description of the file for logging
        
    Returns:
        bool: True if file was processed successfully
    """
    print(f"Waiting for {description} to be processed...")
    
    poll_interval = 2  # seconds to wait between status checks
    max_retries = 30   # maximum number of checks before giving up
    
    for attempt in range(max_retries):
        try:
            file_status = client.files.retrieve(file_id)
            
            if file_status.status == "processed":
                print(f"{description.capitalize()} {file_id} processed successfully.")
                return True
                
            print(f"{description.capitalize()} status: {file_status.status}. Waiting...")
            time.sleep(poll_interval)
            
        except Exception as e:
            print(f"Error checking file status: {e}")
            time.sleep(poll_interval)
    
    print(f"Warning: {description} processing timed out after {max_retries * poll_interval} seconds")
    return False


def create_fine_tuning_job(client, training_file_id, validation_file_id, model="gpt-4o-2024-08-06"):
    """
    Create a fine-tuning job with the specified files.
    
    Args:
        client: OpenAI client instance
        training_file_id (str): ID of the training file
        validation_file_id (str): ID of the validation file
        model (str): Base model to fine-tune
        
    Returns:
        Fine-tuning job object
    """
    print(f"Creating fine-tuning job using model: {model}")
    
    try:
        job = client.fine_tuning.jobs.create(
            training_file=training_file_id,
            validation_file=validation_file_id,
            model=model
        )
        print(f"Fine-tuning job created with ID: {job.id}")
        print(f"You can monitor the job status with: client.fine_tuning.jobs.retrieve(\"{job.id}\")")
        return job
    except Exception as e:
        print(f"Error creating fine-tuning job: {e}")
        raise


def main():
    """Main function to orchestrate the fine-tuning process."""
    # Setup
    script_dir = Path(__file__).parent
    train_file_path = script_dir / "data_jsonl/train.jsonl"
    val_file_path = script_dir / "data_jsonl/val.jsonl"
    
    # Validate file existence
    for file_path in [train_file_path, val_file_path]:
        if not file_path.exists():
            raise FileNotFoundError(f"Required file not found: {file_path}")
    
    # Initialize OpenAI client
    api_key = load_api_key()
    print(f"API key loaded: {api_key[:4]}...{api_key[-4:]}")
    client = OpenAI(api_key=api_key)
    
    # Upload files
    train_file = upload_file(client, train_file_path)
    val_file = upload_file(client, val_file_path)
    
    # Wait for files to be processed
    train_processed = wait_for_file_processing(client, train_file.id, "training file")
    val_processed = wait_for_file_processing(client, val_file.id, "validation file")
    
    if train_processed and val_processed:
        print("Both files processed. Ready to create fine-tuning job.")
        # Create fine-tuning job
        job = create_fine_tuning_job(client, train_file.id, val_file.id)
    else:
        print("File processing issue detected. Fine-tuning job creation aborted.")


if __name__ == "__main__":
    main()