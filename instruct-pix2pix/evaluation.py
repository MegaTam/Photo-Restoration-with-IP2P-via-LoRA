import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as T
import os
import json
import sys
import numpy as np

# --- 1. Configs ---
GENERATED_IMAGES_DIR = "/home/ztanak/MSBD6000Q/instruct-pix2pix/inference_lora"
# Ground-Truth
DATASET_ROOT = "/home/ztanak/MSBD6000Q/instruct-pix2pix/ip2p_lora_dataset/old photo_test"
# Testing data
METADATA_FILE_PATH = os.path.join(DATASET_ROOT, "test.jsonl") 
# IMAGE_RESOLUTION = 512
IMAGE_RESOLUTION = 1024

def _psnr(a: torch.Tensor, b: torch.Tensor, data_range: float = 1.0) -> torch.Tensor:
    mse = F.mse_loss(a, b, reduction='mean')
    mse = torch.clamp(mse, min=1e-10)
    return 10.0 * torch.log10((data_range ** 2) / mse)

def _gaussian_kernel(window_size: int = 11, sigma: float = 1.5, channels: int = 3, device=None, dtype=None):
    coords = torch.arange(window_size, dtype=dtype, device=device) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma * sigma))
    g = (g / g.sum()).unsqueeze(1)
    kernel_2d = (g @ g.t()).unsqueeze(0).unsqueeze(0)  # 1 x 1 x K x K
    kernel = kernel_2d.repeat(channels, 1, 1, 1)       # C x 1 x K x K
    return kernel

def _ssim(a: torch.Tensor, b: torch.Tensor, data_range: float = 1.0, window_size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2
    _, c, _, _ = a.shape
    k = _gaussian_kernel(window_size, sigma, c, a.device, a.dtype)
    pad = window_size // 2
    mu1 = F.conv2d(a, k, groups=c, padding=pad)
    mu2 = F.conv2d(b, k, groups=c, padding=pad)
    mu1_sq, mu2_sq, mu1_mu2 = mu1 * mu1, mu2 * mu2, mu1 * mu2
    sigma1_sq = F.conv2d(a * a, k, groups=c, padding=pad) - mu1_sq
    sigma2_sq = F.conv2d(b * b, k, groups=c, padding=pad) - mu2_sq
    sigma12 = F.conv2d(a * b, k, groups=c, padding=pad) - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


# 3. Helper Function
def load_and_preprocess_image(image_path: str, resolution: int, device: torch.device) -> torch.Tensor:
    """Loading image and transform into tensor ranged in [0, 1]"""
    try:
        image = Image.open(image_path).convert("RGB").resize((resolution, resolution))
        transform = T.Compose([
            T.ToTensor()
        ])
        tensor = transform(image)
        return tensor.unsqueeze(0).to(device)
    except FileNotFoundError:
        print(f"[ERROR] Can not find file: {image_path}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"[ERROR] Fail to load {image_path}: {e}", file=sys.stderr)
        return None


# 4. Main Evaluation Loop
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Starting evaluation...")
    print(f"Generated Images Dir: {GENERATED_IMAGES_DIR}")
    print(f"Ground Truth From: {DATASET_ROOT}")
    print(f"Metadata File: {METADATA_FILE_PATH}")
    
    all_psnr_scores = []
    all_ssim_scores = []

    try:
        with open(METADATA_FILE_PATH, "r") as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"[FATAL ERROR] Can not find METADATA: {METADATA_FILE_PATH}", file=sys.stderr)
        sys.exit(1)
        
    total_images = len(lines)
    print(f"Found {total_images} samples to evaluate.")
    
    for i, line in enumerate(lines):
        try:
            data = json.loads(line)
            # 1. Ground-Truth images
            gt_relative_path = data["output_image_path"]
            gt_full_path = os.path.join(DATASET_ROOT, gt_relative_path)
            # 2. Generated images
            input_relative_path = data["input_image_path"]
            base_filename = os.path.splitext(os.path.basename(input_relative_path))[0]
            generated_filename = f"{base_filename}_restored.png"
            generated_full_path = os.path.join(GENERATED_IMAGES_DIR, generated_filename)
            if i//500 == 0:
                print(f"\n--- Processing image {i+1}/{total_images} ---")
            # 3. Load and preprocess images
            gen_tensor = load_and_preprocess_image(generated_full_path, IMAGE_RESOLUTION, device)
            gt_tensor = load_and_preprocess_image(gt_full_path, IMAGE_RESOLUTION, device)
            # 4. Evaluate
            with torch.no_grad(): 
                psnr_score = _psnr(gen_tensor, gt_tensor, data_range=1.0)
                ssim_score = _ssim(gen_tensor, gt_tensor, data_range=1.0)
            all_psnr_scores.append(psnr_score.item())
            all_ssim_scores.append(ssim_score.item())
            print(f"  > PSNR: {psnr_score.item():.2f} dB")
            print(f"  > SSIM: {ssim_score.item():.4f}")
        except Exception as e:
            print(f"[ERROR] Fail to evaluate image: {data.get('input_image_path')} with error in {e}", file=sys.stderr)

    # --- 5. Final Report ---
    avg_psnr = np.mean(all_psnr_scores)
    avg_ssim = np.mean(all_ssim_scores)
    print("\n--- Final Evaluation Report ---")
    print(f"Total images evaluated: {len(all_psnr_scores)}")
    print(f"Average PSNR: {avg_psnr:.2f} dB")
    print(f"Average SSIM: {avg_ssim:.4f}")

if __name__ == "__main__":
    main()
    