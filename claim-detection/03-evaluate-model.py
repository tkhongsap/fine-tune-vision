import re
import json
import os
import time
import dotenv

from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
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

api_key = load_api_key()
print(f"API key loaded: {api_key[:4]}...{api_key[-4:]}")
client = OpenAI(api_key=api_key)

source_dir = Path(__file__).parent / "data_jsonl"


# load the test data from JSONL file
test_data = []
with open(source_dir / "test.jsonl", "r") as f:
    for line in f:
        test_data.append(json.loads(line))

# Debug: Check the structure of the first test example
if test_data:
    print("First test example keys:", list(test_data[0].keys()))
    print("Sample message structure:", test_data[0].get("messages", [])[0] if test_data[0].get("messages", []) else "No messages")
    
    # Check if messages are properly formatted
    sample_messages = test_data[0].get("messages", [])
    if sample_messages:
        print("\nSample messages format check:")
        for i, msg in enumerate(sample_messages):
            print(f"Message {i} - Role: {msg.get('role')}")
            if isinstance(msg.get('content'), list):
                print(f"  Content is a list with {len(msg['content'])} items")
                for j, content_item in enumerate(msg['content']):
                    print(f"    Item {j} type: {content_item.get('type')}")
            else:
                print(f"  Content is a string: {msg.get('content')[:30]}...")
    
    # Check if any examples have 'answer' key
    has_answer = any('answer' in ex for ex in test_data)
    print(f"Any examples with 'answer' key: {has_answer}")
    # If not, check for potential alternate keys that might contain the answer
    if not has_answer and test_data:
        all_keys = set()
        for ex in test_data[:10]:  # Check first 10 examples
            all_keys.update(ex.keys())
        print("All available keys in first 10 examples:", all_keys)

def process_example(example, model):
    # First, extract the expected answer from the example's messages if it exists
    # (The ground truth would be in the assistant messages in the example itself)
    reference_answer = "UNKNOWN"
    for msg in example["messages"]:
        if msg["role"] == "assistant":
            if isinstance(msg["content"], list):
                # For assistant messages with content as a list
                content_parts = []
                for content_item in msg["content"]:
                    if isinstance(content_item, dict) and content_item.get("type") == "text":
                        content_parts.append(content_item.get("text", ""))
                reference_answer = " ".join(content_parts)
            else:
                reference_answer = msg["content"]
    
    # Prepare messages for the API - ensure format is correct
    api_messages = []
    for msg in example["messages"]:
        role = msg["role"]
        content = msg["content"]
        
        # Skip assistant messages since we're asking for a prediction
        if role == "assistant":
            continue
            
        api_messages.append({"role": role, "content": content})
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=api_messages,  # Use the formatted messages
            store=True,
            metadata={'dataset': 'test'}
        )
        predicted_answer = response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error processing example: {e}")
        predicted_answer = "ERROR"
    
    # regex to get the question ID
    user_msg_content = ""
    for msg in example["messages"]:
        if msg["role"] == "user":
            if isinstance(msg["content"], list) and len(msg["content"]) > 0:
                if isinstance(msg["content"][0], dict) and "text" in msg["content"][0]:
                    user_msg_content = msg["content"][0]["text"]
            else:
                user_msg_content = msg["content"]
                
    match = re.search(r'\[(\d+)\]', user_msg_content)
    if match:
        example_id = int(match.group(1))
    else:
        example_id = -1

    return {
        "example_id": example_id,
        "predicted_answer": predicted_answer,
        "actual_answer": reference_answer
    }

# run the prompts through the finetuned model and store the results
model = "ft:gpt-4o-2024-08-06:pegasus001::BPO8MgOU"
results = []
with ThreadPoolExecutor() as executor:
    futures = {executor.submit(process_example, example, model): example for example in test_data}
    for future in tqdm(as_completed(futures), total=len(futures)):
        results.append(future.result())

# save the results to a file
with open(source_dir / "ft-model-results.jsonl", "w") as f:
    for result in results:
        json.dump(result, f)
        f.write("\n")

# run the prompts through the non-fine-tuned model and store the results
model = "gpt-4o"
results = []
with ThreadPoolExecutor() as executor:
    futures = {executor.submit(process_example, example, model): example for example in test_data}
    for future in tqdm(as_completed(futures), total=len(futures)):
        results.append(future.result())

# save the results to a file
with open(source_dir / "4o-model-results.jsonl", "w") as f:
    for result in results:
        json.dump(result, f)
        f.write("\n")