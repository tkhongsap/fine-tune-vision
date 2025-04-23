from openai import OpenAI, ChatCompletion
import json
import os
import dotenv

dotenv.load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
print(openai_api_key)

client = OpenAI(api_key=openai_api_key)


# upload training file
train_file = client.files.create(
  file=open("claim-detection/data_jsonl/train.jsonl", "rb"),
  purpose="fine-tune"
)

# upload validation file
val_file = client.files.create(
  file=open("claim-detection/data_jsonl/val.jsonl", "rb"),
  purpose="fine-tune"
)

# Wait for the files to be processed before creating the fine-tuning job
print("Waiting for training file to be processed...")
while True:
    train_file_status = client.files.retrieve(train_file.id)
    if train_file_status.status == "processed":
        print(f"Training file {train_file.id} processed successfully.")
        break
    print(f"Training file status: {train_file_status.status}. Waiting...")
    import time
    time.sleep(2)  # Wait for 5 seconds before checking again

print("Waiting for validation file to be processed...")
while True:
    val_file_status = client.files.retrieve(val_file.id)
    if val_file_status.status == "processed":
        print(f"Validation file {val_file.id} processed successfully.")
        break
    print(f"Validation file status: {val_file_status.status}. Waiting...")
    time.sleep(2)  # Wait for 5 seconds before checking again

print("Both files processed. Ready to create fine-tuning job.")


# create fine tuning job
file_train = train_file.id
file_val = val_file.id

client.fine_tuning.jobs.create(
  training_file=file_train,
  # note: validation file is optional
  validation_file=file_val,
  model="gpt-4o-2024-08-06"
)