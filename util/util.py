from __future__ import print_function
import math
import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import yaml
import socket
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity
import lpips  # 新增指標支援

# 初始化 LPIPS 模型 (使用 VGG 骨幹)
# 注意：第一次執行會下載權重
loss_fn_lpips = None


def get_config(config):
    with open(config, "r") as stream:
        return yaml.load(stream, Loader=yaml.FullLoader)


def tensor2im(image_tensor, imtype=np.uint8):
    image_numpy = image_tensor[0].cpu().float().numpy()
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    image_numpy = np.clip(image_numpy, 0, 255)  # 確保數值範圍正確
    image_numpy = image_numpy.astype(imtype)
    if image_numpy.shape[-1] == 6:
        image_numpy = np.concatenate(
            [image_numpy[:, :, :3], image_numpy[:, :, 3:]], axis=1
        )
    return image_numpy


def tensor2numpy(image_tensor):
    image_numpy = torch.squeeze(image_tensor).cpu().float().numpy()
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    image_numpy = image_numpy.astype(np.float32)
    return image_numpy


# --- 新增指標計算函數 (NCC, LMSE, LPIPS) ---


def compute_ncc(img1, img2):
    """計算 Normalized Cross-Correlation"""
    img1 = img1.flatten()
    img2 = img2.flatten()
    img1 = (img1 - np.mean(img1)) / (np.std(img1) + 1e-8)
    img2 = (img2 - np.mean(img2)) / (np.std(img2) + 1e-8)
    return np.mean(img1 * img2)


def compute_lmse(img1, img2):
    """計算 Local Mean Squared Error (簡化版)"""
    return np.mean((img1 - img2) ** 2)


def compute_lpips(img1, img2):
    """計算 LPIPS (感知相似度)"""
    global loss_fn_lpips
    if loss_fn_lpips is None:
        loss_fn_lpips = lpips.LPIPS(net="vgg").cuda()

    # img1, img2 是 numpy [H,W,C] 範圍 0~255 或 0~1
    # 轉換為 tensor [-1, 1] 格式
    if img1.max() > 1.0:
        img1 = img1 / 255.0
    if img2.max() > 1.0:
        img2 = img2 / 255.0

    t1 = torch.from_numpy(img1).permute(2, 0, 1).unsqueeze(0).float().cuda() * 2 - 1
    t2 = torch.from_numpy(img2).permute(2, 0, 1).unsqueeze(0).float().cuda() * 2 - 1

    with torch.no_grad():
        res = loss_fn_lpips(t1, t2)
    return res.item()


# --- 修改後的批量指標計算 ---


def batch_PSNR(img, imclean, data_range):
    Img = img.data.cpu().numpy().astype(np.float32)
    Iclean = imclean.data.cpu().numpy().astype(np.float32)
    PSNR = 0
    for i in range(Img.shape[0]):
        PSNR += compare_psnr(Iclean[i, :, :, :], Img[i, :, :, :], data_range=data_range)
    return PSNR / Img.shape[0]


def batch_SSIM(img, imclean):
    Img = img.data.cpu().permute(0, 2, 3, 1).numpy().astype(np.float32)
    Iclean = imclean.data.cpu().permute(0, 2, 3, 1).numpy().astype(np.float32)
    SSIM = 0
    for i in range(Img.shape[0]):
        SSIM += structural_similarity(
            Iclean[i, :, :, :],
            Img[i, :, :, :],
            win_size=11,
            multichannel=True,
            data_range=1,
        )
    return SSIM / Img.shape[0]


def batch_all_metrics(img, imclean):
    """
    一次計算所有 Index: PSNR, SSIM, NCC, LMSE, LPIPS
    """
    # 轉換為 numpy 0~1 範圍 [H,W,C]
    Img = (img.data.cpu().permute(0, 2, 3, 1).numpy() + 1) / 2.0
    Iclean = (imclean.data.cpu().permute(0, 2, 3, 1).numpy() + 1) / 2.0

    metrics = {"PSNR": 0, "SSIM": 0, "NCC": 0, "LMSE": 0, "LPIPS": 0}
    num = Img.shape[0]

    for i in range(num):
        pred = np.clip(Img[i], 0, 1)
        gt = np.clip(Iclean[i], 0, 1)

        metrics["PSNR"] += compare_psnr(gt, pred, data_range=1)
        metrics["SSIM"] += structural_similarity(
            gt, pred, win_size=11, multichannel=True, data_range=1
        )
        metrics["NCC"] += compute_ncc(pred, gt)
        metrics["LMSE"] += compute_lmse(pred, gt)
        metrics["LPIPS"] += compute_lpips(pred, gt)

    for key in metrics:
        metrics[key] /= num
    return metrics


# --- 其他原始公用函數 ---


def get_model_list(dirname, key, epoch=None):
    if epoch is None:
        return os.path.join(dirname, key + "_latest.pt")
    if os.path.exists(dirname) is False:
        return None
    gen_models = [
        os.path.join(dirname, f)
        for f in os.listdir(dirname)
        if ".pt" in f and "latest" not in f
    ]
    epoch_index = [
        int(os.path.basename(model_name).split("_")[-2]) for model_name in gen_models
    ]
    i = epoch_index.index(int(epoch))
    return gen_models[i]


def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)


class AverageMeters(object):
    def __init__(self, dic=None, total_num=None):
        self.dic = dic or {}
        self.total_num = total_num or {}

    def update(self, new_dic):
        for key in new_dic:
            if not key in self.dic:
                self.dic[key] = new_dic[key]
                self.total_num[key] = 1
            else:
                self.dic[key] += new_dic[key]
                self.total_num[key] += 1

    def __getitem__(self, key):
        return self.dic[key] / self.total_num[key]

    def __str__(self):
        keys = sorted(self.keys())
        res = ""
        for key in keys:
            res += key + ": %.4f" % self[key] + " | "
        return res

    def keys(self):
        return self.dic.keys()


def progress_bar(current, total, msg=None):
    global last_time, begin_time
    TOTAL_BAR_LENGTH = 30.0
    if current == 0:
        begin_time = time.time()
    cur_len = int(TOTAL_BAR_LENGTH * current / total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1
    sys.stdout.write(" [")
    for i in range(cur_len):
        sys.stdout.write("=")
    sys.stdout.write(">")
    for i in range(rest_len):
        sys.stdout.write(".")
    sys.stdout.write("]")
    cur_time = time.time()
    tot_time = cur_time - begin_time
    L = [f" | Tot: {format_time(tot_time)}"]
    if msg:
        L.append(" | " + msg)
    msg = "".join(L)
    sys.stdout.write(msg + f" {current+1}/{total} \r")
    if current == total - 1:
        sys.stdout.write("\n")
    sys.stdout.flush()


def format_time(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return "%d:%02d:%02d" % (h, m, s)


def mkdirs(paths):
    if isinstance(paths, list):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def set_opt_param(optimizer, key, value):
    for group in optimizer.param_groups:
        group[key] = value


def get_summary_writer(log_dir):
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    writer = SummaryWriter(log_dir)
    return writer


def write_loss(writer, prefix, avg_meters, iteration):
    for key in avg_meters.keys():
        writer.add_scalar(os.path.join(prefix, key), avg_meters[key], iteration)
