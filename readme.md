# Old Photo Restoration with InstructPix2Pix via Lightweight LoRA Fine-Tuning

**Course:** MSBD6000Q - Vision Language Models for Vision Tasks  
**Project:** Final Project  
**Author:** TAN Zhaohong, CHENG Ruiyi, XIONG Yuchen

## 1. Project Overview

This project addresses the challenge of restoring old, degraded photographs (specifically removing scratches and noise) while preserving their original structure and historical texture. 

We utilize **InstructPix2Pix (IP2P)**, a text-guided image editing model, and fine-tune it using **Low-Rank Adaptation (LoRA)**. By injecting low-rank matrices into the attention layers of the U-Net, we adapt the general-purpose editing model to the specific task of old photo restoration ("Remove the noise from the picture") with minimal computational cost.

**Key Features:**
* **Lightweight Fine-tuning:** Uses LoRA (Rank 8/16) instead of full model retraining.
* **Text-Guided Restoration:** Controlled via the prompt "Remove the noise from the picture".
* **Evaluation:** Quantitative assessment using PSNR and SSIM metrics.

---

## 2. Repository Structure

The project relies on specific absolute paths configured on the server.

* **Project Root:** `/home/ztanak/MSBD6000Q`
* **Diffusers Library (Training Script):** `~/MSBD6000Q/diffusers/examples/research_projects/instructpix2pix_lora`
* **IP2P Directory (Inference & Data):** `~/MSBD6000Q/instruct-pix2pix`
* **Dataset:** `~/MSBD6000Q/instruct-pix2pix/ip2p_lora_dataset`
* **Checkpoints:** `~/MSBD6000Q/instruct-pix2pix/checkpoints`


## 3. Environment Setup
The project runs on a Linux environment with CUDA 11.8.
### 3.1 Basic Server Commands
```bash
# Check current disk usage
ncdu ~
# Check GPU usage
squota -S 2025-09-01
```
### 3.2 Prerequisites & Installation
Ensure Anaconda3 and CUDA 11.8 modules are loaded.
```bash
# Load modules
module load Anaconda3/2023.09-0
module load cuda11.8/toolkit/11.8.0
module load cudnn8.6-cuda11.8/8.6.0.16

# Verify NVCC version
nvcc --version

# Create and activate environment
conda create -n ip2p_lora_env python=3.10 -y
source activate ip2p_lora_env

# Install PyTorch for CUDA 11.8
pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118)

# Install Diffusers and LoRA dependencies
pip install diffusers transformers accelerate peft
```

## 4. Data Preparation
We use the SynOld dataset. The dataset is organized into input_images (LQ) and output_images (GT).
### 4.1 Dataset Location
Path: /home/ztanak/MSBD6000Q/instruct-pix2pix/ip2p_lora_dataset  
Train Set: old photo_train/  
Test Set: old photo_test/
### 4.2 Metadata Generation
We use generate_metadata.py to create .jsonl files required for training. The script maps input images to output images with the instruction "Remove the noise from the picture".
```bash
# Generate Train Metadata
python ~/MSBD6000Q/instruct-pix2pix/generate_metadata.py \
    --lq_dir "/home/ztanak/MSBD6000Q/instruct-pix2pix/ip2p_lora_dataset/old photo_train/LQ" \
    --gt_dir "/home/ztanak/MSBD6000Q/instruct-pix2pix/ip2p_lora_dataset/old photo_train/GT" \
    --output_file "/home/ztanak/MSBD6000Q/instruct-pix2pix/ip2p_lora_dataset/old photo_train/train.jsonl" \
    --instruction "Remove the noise from the picture"

# Generate Test Metadata
python ~/MSBD6000Q/instruct-pix2pix/generate_metadata.py \
    --lq_dir "/home/ztanak/MSBD6000Q/instruct-pix2pix/ip2p_lora_dataset/old photo_test/LQ" \
    --gt_dir "/home/ztanak/MSBD6000Q/instruct-pix2pix/ip2p_lora_dataset/old photo_test/GT" \
    --output_file "/home/ztanak/MSBD6000Q/instruct-pix2pix/ip2p_lora_dataset/old photo_test/test.jsonl" \
    --instruction "Remove the noise from the picture"
```

## 5. Training (LoRA Fine-Tuning)
We fine-tune the model using accelerate.  
Base Model: timbrooks/instruct-pix2pix  
Script Location: ~/MSBD6000Q/diffusers/examples/research_projects/instructpix2pix_lora  
### Training Command
```bash
# Activate environment
module load Anaconda3/2023.09-0
source activate ip2p_lora_env

# Navigate to script directory
cd ~/MSBD6000Q/diffusers/examples/research_projects/instructpix2pix_lora

# Launch training
accelerate launch train_instruct_pix2pix_lora.py \
    --pretrained_model_name_or_path="timbrooks/instruct-pix2pix" \
    --dataset_name="/home/ztanak/MSBD6000Q/instruct-pix2pix/ip2p_lora_dataset/old photo_train" \
    --original_image_column="input_image_path" \
    --edited_image_column="output_image_path" \
    --edit_prompt_column="instruction" \
    --output_dir="/home/ztanak/MSBD6000Q/instruct-pix2pix/checkpoints" \
    --resolution=512 \
    --rank=8 \
    --learning_rate=1e-4 \
    --train_batch_size=4 \
    --max_train_steps=5000 \
    --checkpointing_steps=500 \
    --seed=42
```
Training Outputs: The LoRA weights are saved at: /home/ztanak/MSBD6000Q/instruct-pix2pix/checkpoints/pytorch_lora_weights.safetensors  

## 6. Inference
We run inference using a custom script that loads the base IP2P model and attaches the trained LoRA weights.  
Inference Script: ~/MSBD6000Q/instruct-pix2pix/inference.py  
Output Location: ~/MSBD6000Q/instruct-pix2pix/inference_lora/
### Running Inference
```bash
# Run the inference script
python ~/MSBD6000Q/instruct-pix2pix/inference.py
```

## 7. Evaluation & Results
We evaluate the model performance using PSNR (Peak Signal-to-Noise Ratio) and SSIM (Structural Similarity Index) on the test set (200 images).
### Run Evaluation
We use a custom script to calculate the metrics by comparing the generated images (in `inference_lora`) against the ground truth (GT).
```bash
# Run the evaluation script
python ~/MSBD6000Q/instruct-pix2pix/evaluation.py
```


## 8. Reference
1. InstructPix2Pix: Tim Brooks, et al. "InstructPix2Pix: Learning to Follow Image Editing Instructions." CVPR 2023.  
2. LoRA: Edward J. Hu, et al. "LoRA: Low-Rank Adaptation of Large Language Models." ICLR 2022.
3. SynOld Dataset: https://github.com/wushunshun/SynOld