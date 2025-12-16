import torch
import util.util as util
from models import make_model
import time
import os
import sys
from os.path import join
import math  # <-- [新增] 導入 math 模組
from util.visualizer import Visualizer
from engine import Engine
from options.net_options.train_options import TrainOptions
from tools import mutils
import data.sirs_dataset as datasets
from data.image_folder import read_fns


# --- 客製化儲存參數 START ---
# 總訓練圖片數 N_train = 14288 * 0.8 = 11430 (估計值)
# 確保這些參數在程式碼被執行的 Colab 環境中是正確的
BATCH_SIZE_CONST = 1
SAVE_IMAGES_INTERVAL = 1500  # 您要求的儲存頻率 (張)

# 計算儲存間隔 (Iteration/Batch 數)
SAVE_ITER_FREQ = math.ceil(SAVE_IMAGES_INTERVAL / BATCH_SIZE_CONST)

# Google Drive 模型儲存路徑 (必須是 Colab 環境可存取到的絕對路徑)
MODEL_SAVE_PATH = "/content/drive/MyDrive/Colab Notebooks/期末考/DSRNet/"
# --- 客製化儲存參數 END ---


opt = TrainOptions().parse()
print(opt)

# 將常數賦值給 opt.batchSize，以確保程式邏輯使用一致的 Batch Size
opt.batchSize = BATCH_SIZE_CONST

if opt.debug:
    opt.display_id = 1
    opt.display_freq = 1
    opt.print_freq = 20
    opt.nEpochs = 40
    opt.max_dataset_size = 9999
    opt.no_log = False
    opt.nThreads = 0
    opt.decay_iter = 0
    opt.serial_batches = True
    opt.no_flip = True

# modify the following code to
datadir = os.path.join(opt.base_dir)

datadir_1_input = join(datadir, "training set/training set 1_13700/syn")
datadir_1_target = join(datadir, "training set/training set 1_13700/t")
datadir_2_input = join(datadir, "training set/training set 2_Berkeley_Real/blended")
datadir_2_target = join(
    datadir, "training set/training set 2_Berkeley_Real/transmission_layer"
)
datadir_3_input = join(datadir, "training set/training set 3_Nature/blended")
datadir_3_target = join(
    datadir, "training set/training set 3_Nature/transmission_layer"
)
datadir_4_input = join(
    datadir, "training set/training set 4_unaligned_train250/blended"
)
datadir_4_target = join(
    datadir, "training set/training set 4_unaligned_train250/transmission_layer"
)

train_dataset_1 = datasets.DSRDataset(
    datadir_1_input,
    fns=None,  # 假設 DSRDataset 第二個參數是 Target Path
    size=opt.max_dataset_size,
    enable_transforms=True,
)

train_dataset_2 = datasets.DSRDataset(
    datadir_2_input,
    fns=None,  # 假設 DSRDataset 第二個參數是 Target Path
    size=opt.max_dataset_size,
    enable_transforms=True,
)

train_dataset_3 = datasets.DSRDataset(
    datadir_3_input,
    fns=None,  # 假設 DSRDataset 第3個參數是 Target Path
    size=opt.max_dataset_size,
    enable_transforms=True,
)

train_dataset_4 = datasets.DSRDataset(
    datadir_4_input,
    fns=None,  # 假設 DSRDataset 第4個參數是 Target Path
    size=opt.max_dataset_size,
    enable_transforms=True,
)


train_dataset_fusion = datasets.FusionDataset(
    [train_dataset_1, train_dataset_2, train_dataset_3, train_dataset_4],
    [0.25, 0.25, 0.25, 0.25],
)

train_dataloader_fusion = datasets.DataLoader(
    train_dataset_fusion,
    batch_size=opt.batchSize,
    shuffle=not opt.serial_batches,
    pin_memory=True,
    prefetch_factor=32,
    num_workers=0,
)

""" 註解eval函數
eval_dataset_real = datasets.DSRTestDataset(
    join(datadir, f"test/real20_{opt.real20_size}"),
    fns=read_fns("data/real_test.txt"),
    if_align=opt.if_align,
)
eval_dataset_solidobject = datasets.DSRTestDataset(
    join(datadir, "test/SIR2/SolidObjectDataset"), if_align=opt.if_align
)
eval_dataset_postcard = datasets.DSRTestDataset(
    join(datadir, "test/SIR2/PostcardDataset"), if_align=opt.if_align
)
eval_dataset_wild = datasets.DSRTestDataset(
    join(datadir, "test/SIR2/WildSceneDataset"), if_align=opt.if_align
)

eval_dataloader_real = datasets.DataLoader(
    eval_dataset_real,
    batch_size=1,
    shuffle=False,
    pin_memory=True,
    prefetch_factor=32,
    num_workers=32,
)

eval_dataloader_solidobject = datasets.DataLoader(
    eval_dataset_solidobject,
    batch_size=1,
    shuffle=False,
    pin_memory=True,
    prefetch_factor=32,
    num_workers=32,
)
eval_dataloader_postcard = datasets.DataLoader(
    eval_dataset_postcard,
    batch_size=1,
    shuffle=False,
    pin_memory=True,
    prefetch_factor=32,
    num_workers=32,
)

eval_dataloader_wild = datasets.DataLoader(
    eval_dataset_wild,
    batch_size=1,
    shuffle=False,
    pin_memory=True,
    prefetch_factor=32,
    num_workers=32,
)
"""

"""Main Loop"""
engine = Engine(opt)
result_dir = os.path.join(
    f"./checkpoints/{opt.name}/results", mutils.get_formatted_time()
)


def set_learning_rate(lr):
    for optimizer in engine.model.optimizers:
        print("[i] set learning rate to {}".format(lr))
        util.set_opt_param(optimizer, "lr", lr)


if opt.resume or opt.debug_eval:
    save_dir = os.path.join(result_dir, "%03d" % engine.epoch)
    os.makedirs(save_dir, exist_ok=True)
    engine.save_model()
"""註解eval
    engine.eval(
        eval_dataloader_real,
        dataset_name="testdata_real20",
        savedir=save_dir,
        suffix="real20",
        max_save_size=10,
    )
    engine.eval(
        eval_dataloader_solidobject,
        dataset_name="testdata_solidobject",
        savedir=save_dir,
        suffix="solidobject",
        max_save_size=10,
    )
    engine.eval(
        eval_dataloader_postcard,
        dataset_name="testdata_postcard",
        savedir=save_dir,
        suffix="postcard",
        max_save_size=10,
    )
    engine.eval(
        eval_dataloader_wild,
        dataset_name="testdata_wild",
        savedir=save_dir,
        suffix="wild",
        max_save_size=10,
    )
"""

# define training strategy
set_learning_rate(opt.lr)
while engine.epoch < 50:
    print("random_seed: ", opt.seed)

    # [修改處] 呼叫 engine.train 時，傳入客製化儲存所需的參數
    engine.train(
        train_dataloader_fusion,
        custom_save_path=MODEL_SAVE_PATH,
        custom_save_freq=SAVE_ITER_FREQ,
        batch_size=opt.batchSize,
    )

    if engine.epoch % 1 == 0:
        save_dir = os.path.join(result_dir, "%03d" % engine.epoch)
        os.makedirs(save_dir, exist_ok=True)
        
        #     engine.eval(
        #     eval_dataloader_real,
        #     dataset_name="testdata_real20",
        #     savedir=save_dir,
        #     suffix="real20",
        #     max_save_size=10,
        # )
        # engine.eval(
        #     eval_dataloader_solidobject,
        #     dataset_name="testdata_solidobject",
        #     savedir=save_dir,
        #     suffix="solidobject",
        #     max_save_size=10,
        # )
        # engine.eval(
        #     eval_dataloader_postcard,
        #     dataset_name="testdata_postcard",
        #     savedir=save_dir,
        #     suffix="postcard",
        #     max_save_size=10,
        # )
        # engine.eval(
        #     eval_dataloader_wild,
        #     dataset_name="testdata_wild",
        #     savedir=save_dir,
        #     suffix="wild",
        #     max_save_size=10,
        # )

