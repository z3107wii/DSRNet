import os
import time
import torch
import pandas as pd  # 建議安裝以方便處理表格
from os.path import join
import torch.backends.cudnn as cudnn
from ptflops import get_model_complexity_info

import data.sirs_dataset as datasets
from engine import Engine
from options.net_options.train_options import TrainOptions
from tools import mutils
import util.util as util

# --- 環境與參數設定 ---
opt = TrainOptions().parse()
opt.isTrain = False
cudnn.benchmark = True
opt.no_log = True
opt.display_id = 0
opt.verbose = False

# 強制指定您的權重路徑
opt.weight_path = "/content/drive/MyDrive/Colab Notebooks/Term_Project/DSRNet/checkpoints/dsrnet_l/dsrnet_l_DSRNet_3000.pth"

# 初始化 Engine
engine = Engine(opt)

# --- 效能數據 (圖二): Parameters & FLOPs ---
print("\n" + "=" * 50)
print("📊 正在計算模型效能數據 (Table 2)...")
with torch.cuda.device(0):
    macs, params = get_model_complexity_info(
        engine.model.netG, (3, 224, 224), as_strings=True, print_per_layer_stat=False
    )
    perf_msg = f"Parameters: {params} | FLOPs: {macs}"
    print(perf_msg)
print("=" * 50 + "\n")

# --- 定義 7 個測試資料集路徑 ---
# 這些路徑嚴格對應您的雲端資料夾結構
test_base = "/content/drive/MyDrive/Colab Notebooks/Term_Project/testing set"

test_configs = {
    "Berkeley real20_420": join(test_base, "Berkeley real20_420"),
    "CEILNet_real45": join(test_base, "CEILNet_real45"),
    "NRD": join(test_base, "Natural Reflection Dataset(NRD)"),
    "Nature": join(test_base, "Nature"),
    "SIR2_Postcard": join(test_base, "SIR2/PostcardDataset"),
    "SIR2_SolidObject": join(test_base, "SIR2/SolidObjectDataset"),
    "SIR2_WildScene": join(test_base, "SIR2/WildSceneDataset"),
}

# --- 結果輸出設定 ---
result_save_path = "/content/drive/MyDrive/Colab Notebooks/Term_Project/DSRNet/checkpoints/dsrnet_l/DSRNet_result.txt"
report_file = open(result_save_path, "w", encoding="utf-8")
report_file.write(f"DSRNet Test Report - {mutils.get_formatted_time()}\n")
report_file.write(f"Model Weight: {opt.weight_path}\n")
report_file.write(f"Model Info: {perf_msg}\n")
report_file.write("=" * 80 + "\n")
report_file.write(
    f"{'Dataset':<20} | {'PSNR':<8} | {'SSIM':<8} | {'NCC':<8} | {'LMSE':<8} | {'LPIPS':<8} | {'Time':<8}\n"
)
report_file.write("-" * 80 + "\n")

"""Main Testing Loop"""
for label, path in test_configs.items():
    if not os.path.exists(path):
        print(f"⚠️ 跳過 {label}: 找不到路徑 {path}")
        continue

    print(f"\n🚀 正在評估資料集: {label}")

    # 根據資料集特性載入 (NRD 的資料夾結構與其他不同，需特別處理)
    # NRD 使用 NCCU_I (測試) 與 NCCU_T (驗證)
    if "NRD" in label:
        eval_dataset = datasets.DSRTestDataset(path, if_align=opt.if_align)
        # 註：需確保 datasets.DSRTestDataset 內部有對應 NCCU_I/T 的邏輯
    else:
        eval_dataset = datasets.DSRTestDataset(path, if_align=opt.if_align)

    eval_dataloader = datasets.DataLoader(
        eval_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=opt.nThreads,
        pin_memory=True,
    )

    # 執行評估
    # engine.eval 內部會呼叫之前在 util.py 改好的 batch_all_metrics
    start_time = time.time()
    avg_meters = engine.eval(eval_dataloader, dataset_name=label)
    end_time = time.time()

    avg_run_time = (end_time - start_time) / len(eval_dataloader)

    # 寫入文字檔報告
    line = f"{label:<20} | {avg_meters['PSNR']:<8.4f} | {avg_meters['SSIM']:<8.4f} | {avg_meters['NCC']:<8.4f} | {avg_meters['LMSE']:<8.4f} | {avg_meters['LPIPS']:<8.4f} | {avg_run_time:<8.4f}s\n"
    report_file.write(line)
    report_file.flush()  # 即時存檔防止斷線

    print(f"✅ {label} 指標已存入報告")

report_file.write("=" * 80 + "\n")
report_file.close()

print(f"\n✨ 所有測試完成！結果報告已輸出至: {result_save_path}")
