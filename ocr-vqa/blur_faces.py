import json
import base64
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
from tqdm import tqdm
import os

# Output directories for blurred files
OUTPUT_DIR = "./blurred_files"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Input JSONL files
TRAIN_FILE = "ocr-vqa-train.jsonl"
VAL_FILE = "ocr-vqa-validation.jsonl"
TEST_FILE = "ocr‑vqa‑test.jsonl"  # Note the special dash character

# Setup face detection
FACE_DET = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def decode_base64_to_pil(base64_string):
    """Convert base64 string to PIL Image"""
    img_data = base64.b64decode(base64_string)
    return Image.open(BytesIO(img_data))

def encode_pil_to_base64(pil_img, quality=50):
    """Convert PIL Image to base64 string"""
    buffered = BytesIO()
    pil_img.save(buffered, format="JPEG", quality=quality)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def detect_faces(pil_img, min_size=(30, 30)):
    """Detect faces in PIL Image with more sensitive parameters"""
    gray = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2GRAY)
    faces = FACE_DET.detectMultiScale(
        gray, 
        scaleFactor=1.05,  # More sensitive detection
        minNeighbors=3,    # Require fewer neighboring detections
        minSize=min_size
    )
    return faces

def blur_faces(pil_img, faces):
    """Blur detected faces in the image"""
    img = np.array(pil_img)
    for (x, y, w, h) in faces:
        # Apply heavier blur
        roi = img[y:y+h, x:x+w]
        roi = cv2.GaussianBlur(roi, (51, 51), 50)  # Stronger blur
        img[y:y+h, x:x+w] = roi
    return Image.fromarray(img)

def process_jsonl_file(input_file, output_file):
    """Process a JSONL file, blurring all faces in images"""
    print(f"Processing {input_file}...")
    
    processed_examples = 0
    blurred_examples = 0
    
    with open(input_file, 'r', encoding='utf-8') as infile:
        json_data = []
        
        for line in tqdm(infile):
            try:
                example = json.loads(line.strip())
                
                # Process the messages
                for message in example["messages"]:
                    if message["role"] == "user":
                        for i, content_item in enumerate(message["content"]):
                            if content_item.get("type") == "image_url":
                                # Extract base64 image data
                                image_url = content_item["image_url"]["url"]
                                if image_url.startswith("data:image/jpeg;base64,"):
                                    base64_data = image_url.split("data:image/jpeg;base64,")[1]
                                    
                                    # Decode to PIL image
                                    pil_img = decode_base64_to_pil(base64_data)
                                    
                                    # Detect faces
                                    faces = detect_faces(pil_img)
                                    
                                    # If faces detected, blur them
                                    if len(faces) > 0:
                                        blurred_examples += 1
                                        pil_img = blur_faces(pil_img, faces)
                                    
                                    # Re-encode to base64
                                    new_base64 = encode_pil_to_base64(pil_img, quality=50)
                                    
                                    # Update the content with blurred image
                                    message["content"][i]["image_url"]["url"] = f"data:image/jpeg;base64,{new_base64}"
                
                # Add processed example to output
                json_data.append(example)
                processed_examples += 1
                
            except Exception as e:
                print(f"Error processing an example: {e}")
                continue
        
        # Write output file
        with open(output_file, 'w', encoding='utf-8') as outfile:
            for item in json_data:
                json.dump(item, outfile)
                outfile.write('\n')
    
    print(f"Processed {processed_examples} examples, blurred faces in {blurred_examples} examples")
    return processed_examples, blurred_examples

# Process all three files
def main():
    results = {}
    for input_file, file_type in [
        (TRAIN_FILE, "train"), 
        (VAL_FILE, "validation"), 
        (TEST_FILE, "test")
    ]:
        output_file = os.path.join(OUTPUT_DIR, f"blurred-ocr-vqa-{file_type}.jsonl")
        processed, blurred = process_jsonl_file(input_file, output_file)
        results[file_type] = {"processed": processed, "blurred": blurred}

    # Print summary
    print("\nProcessing Summary:")
    for file_type, counts in results.items():
        print(f"{file_type.capitalize()}: Processed {counts['processed']} examples, blurred {counts['blurred']} images")

if __name__ == "__main__":
    main()