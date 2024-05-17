import torch
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login
import os
import logging

logging.basicConfig(level=logging.INFO)

# Log in to Hugging Face
api_token = "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"
login(api_token)

model_name = 'meta-llama/Llama-2-13b-chat-hf'

print("Starting model loading...")

try:
    # Load the model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        #force_download=True,
        cache_dir="llama"
    )
    print("Model loaded successfully.")

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    print("Tokenizer loaded successfully.")

    # Set up the pipeline
    generator = pipeline(
        model=model, tokenizer=tokenizer,
        task='text-generation',
        temperature=0.001,
        max_new_tokens=500,
        repetition_penalty=1.1
    )
    print("Pipeline created successfully.")

    # Define the prompt
    prompt = "Which emotion do we convey in next sentence: You are such a jerk, asshole! Answer with one word."
    res = generator(prompt)
    print(res[0]["generated_text"])

except Exception as e:
    print(f"An error occurred: {e}")