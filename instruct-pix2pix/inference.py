import torch
from diffusers import StableDiffusionInstructPix2PixPipeline
from PIL import Image
import os
import json # 
import sys # 

# --- 1. Configs ---
MODEL_ID = "timbrooks/instruct-pix2pix"
LORA_WEIGHTS_PATH = "/home/ztanak/MSBD6000Q/instruct-pix2pix/checkpoints/pytorch_lora_weights.safetensors"
DATASET_ROOT = "/home/ztanak/MSBD6000Q/instruct-pix2pix/ip2p_lora_dataset/old photo_test"
# DATASET_ROOT = "/home/ztanak/MSBD6000Q/instruct-pix2pix/ip2p_lora_dataset/cheng"
METADATA_FILE_PATH = os.path.join(DATASET_ROOT, "test.jsonl") 
OUTPUT_DIR = "/home/ztanak/MSBD6000Q/instruct-pix2pix/inference_lora"

# --- 2. Loading model ---
print(f"Loading base model: {MODEL_ID}")
pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    safety_checker=None
)
pipe = pipe.to("cuda")

# --- 3. Loading LoRA weights ---
print(f"Loading LoRA weights from: {LORA_WEIGHTS_PATH}")
try:
    pipe.load_lora_weights(LORA_WEIGHTS_PATH)
    print("LoRA weights loaded successfully.")
except Exception as e:
    print(f"[ERROR] Failed to load LoRA weights: {e}")
# set random seed
generator = torch.Generator("cuda").manual_seed(42)


# --- 4. Loading datasets ---
print(f"\nStarting multi-sample inference...")
print(f"Reading metadata from: {METADATA_FILE_PATH}")
try:
    with open(METADATA_FILE_PATH, "r") as f:
        lines = f.readlines()
except FileNotFoundError:
    print(f"[FATAL ERROR] Metadata file not found: {METADATA_FILE_PATH}")
    sys.exit(1)

total_images = len(lines)
print(f"Found {total_images} images to process.")

for i, line in enumerate(lines):
    
    data = json.loads(line)
    relative_path = data["input_image_path"]
    prompt = data["instruction"]
    
    relative_path = relative_path.strip()
    prompt = prompt.strip()
    
    input_path_full = os.path.join(DATASET_ROOT, relative_path)
    base_filename = os.path.splitext(os.path.basename(relative_path))[0]
    output_filename = f"{base_filename}_restored.png"
    output_path_full = os.path.join(OUTPUT_DIR, output_filename)
    
    print(f"\n--- Processing image {i+1}/{total_images} ---")
    # print(f"Input: {input_path_full}")
    
    # --- 5. Input images ---
    input_image = Image.open(input_path_full).convert("RGB")
    input_image = input_image.resize((1024, 1024))
    # input_image = input_image.resize((512, 512))
    # --- 6. Inference ---
    # print(f"Prompt: '{prompt}'")
    image = pipe(
        prompt=prompt,
        image=input_image,
        num_inference_steps=50,
        image_guidance_scale=1.5,
        guidance_scale=7.5,
        generator=generator
    ).images[0]

    # --- 7. Save Results ---
    image.save(output_path_full)
    # print(f"Success! Restored image saved to: {output_path_full}")
    

print(f"\n--- Multi-sample inference complete ---")
print(f"All results saved in: {OUTPUT_DIR}")
