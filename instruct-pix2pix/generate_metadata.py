import os
import glob
import json
import argparse
from tqdm import tqdm # 用於顯示進度條

def create_metadata(lq_dir, gt_dir, output_file, instruction):
    """
    掃描 LQ 和 GT 目錄，創建 .jsonl 元數據檔案。
    
    假設命名規則為：
    LQ 資料夾中的 'X_LQ.jpg' 對應
    GT 資料夾中的 'X_GT.jpg' 或 'X_GT.png'
    """
    
    print(f"正在掃描 LQ 資料夾: {lq_dir}")
    print(f"正在掃描 GT 資料夾: {gt_dir}")
    
    # 獲取所有 LQ 圖片的絕對路徑
    lq_files = glob.glob(os.path.join(lq_dir, "*_LQ.*"))
    if not lq_files:
        print(f"[錯誤] 在 {lq_dir} 中找不到任何 '_LQ' 檔案。請檢查您的路徑和命名。")
        return

    # 獲取輸出檔案的根目錄，用於生成相對路徑
    # 我們假設 .jsonl 檔案將與 LQ/ 和 GT/ 資料夾位於同一級別
    output_root = os.path.dirname(output_file)

    missing_gt_count = 0
    total_pairs = 0

    with open(output_file, "w") as f:
        # 使用 tqdm 顯示進度
        for lq_full_path in tqdm(lq_files, desc=f"Processing {os.path.basename(output_file)}"):
            
            # 1. 從 LQ 路徑獲取基礎名稱和編號
            # 例如: /.../LQ/8_LQ.jpg -> 8_LQ
            lq_basename = os.path.splitext(os.path.basename(lq_full_path))[0]
            
            # 例如: 8_LQ -> 8
            number_prefix = lq_basename.split('_')[0]
            
            # 2. 構建並查找對應的 GT 檔案
            # 我們使用 glob.glob 查找，以處理不同的副檔名 (jpg, png, etc.)
            gt_search_pattern = os.path.join(gt_dir, f"{number_prefix}_GT.*")
            gt_files_found = glob.glob(gt_search_pattern)
            
            if not gt_files_found:
                # print(f"警告: 找不到 {lq_full_path} 對應的 GT 檔案 (應為 {gt_search_pattern})")
                missing_gt_count += 1
                continue
            
            # 3. 獲取 GT 檔案路徑
            gt_full_path = gt_files_found[0] # 取第一個匹配項
            
            # 4. 創建相對於 .jsonl 檔案的相對路徑
            # 例如: LQ/8_LQ.jpg
            lq_relative_path = os.path.relpath(lq_full_path, output_root) 
            # 例如: GT/8_GT.jpg
            gt_relative_path = os.path.relpath(gt_full_path, output_root)
            
            # 5. 創建 JSON 條目
            data_entry = {
                "input_image_path": lq_relative_path.replace(os.path.sep, '/'), # 確保使用 / 作為路徑分隔符
                "output_image_path": gt_relative_path.replace(os.path.sep, '/'),
                "instruction": instruction
            }
            
            # 6. 寫入檔案
            f.write(json.dumps(data_entry) + "\n")
            total_pairs += 1

    print(f"\n--- 報告 ---")
    print(f"成功寫入 {total_pairs} 條數據到 {output_file}")
    print(f"跳過了 {missing_gt_count} 張找不到對應 GT 的 LQ 圖片。")

def main():
    parser = argparse.ArgumentParser(description="為 IP2P LoRA 訓練生成元數據 .jsonl 檔案")
    parser.add_argument("--lq_dir", type=str, required=True, help="包含低質量 (Input) 圖片的資料夾路徑")
    parser.add_argument("--gt_dir", type=str, required=True, help="包含高質量 (Ground Truth) 圖片的資料夾路徑")
    parser.add_argument("--output_file", type=str, required=True, help="要生成的 .jsonl 檔案的完整路徑")
    parser.add_argument("--instruction", type=str, default="Remove the noise from the picture", help="用於訓練的文本指令")
    
    args = parser.parse_args()
    
    # 確保輸出目錄存在
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    
    create_metadata(args.lq_dir, args.gt_dir, args.output_file, args.instruction)

if __name__ == "__main__":
    main()