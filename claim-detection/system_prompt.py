"""
System prompt definition for the fine-tuning model.
This provides instructions to the model on how to process and respond to questions about bottle images.
"""

SYSTEM_PROMPT = """
You are an assistant that decides whether a bottle can be returned for a deposit refund.
Look at the image and answer with exactly one word: “claimable” or “non-claimable”.
"""

def get_system_prompt():
    """
    Returns the system prompt for the OCR-VQA fine-tuning model.
    
    Returns:
        str: The system prompt text
    """
    return SYSTEM_PROMPT 