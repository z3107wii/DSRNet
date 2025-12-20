import os
import torch
import util.util as util


class BaseModel:
    def name(self):
        return self.__class__.__name__.lower()

    def initialize(self, opt):
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain
        self.Tensor = torch.cuda.FloatTensor if self.gpu_ids else torch.Tensor

        # --- 核心修正：路徑清理邏輯 ---
        # 直接使用傳入的 checkpoints_dir 和 name，避免疊加兩次絕對路徑
        # 如果 opt.checkpoints_dir 已經是絕對路徑，os.path.join 會正確處理
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)
        self.model_save_dir = self.save_dir

        # 確保資料夾存在
        if not os.path.exists(self.model_save_dir):
            os.makedirs(self.model_save_dir, exist_ok=True)

        self._count = 0

    def set_input(self, input):
        self.input = input

    def forward(self, mode="train"):
        pass

    def test(self):
        pass

    def get_image_paths(self):
        pass

    def optimize_parameters(self):
        pass

    def get_current_visuals(self):
        return self.input

    def get_current_errors(self):
        return {}

    def print_optimizer_param(self):
        print(self.optimizers[-1])

    def save(self, label=None):
        # 獲取當前訓練進度，如果沒有定義則預設為 0
        epoch = getattr(self, "epoch", 0)
        iterations = getattr(self, "iterations", 0)

        # 確保目錄存在
        os.makedirs(self.model_save_dir, exist_ok=True)

        # 檔名格式修正為 .pth (PyTorch 慣用格式)
        if label is None:
            filename = "%s_epoch%03d_it%08d.pth" % (self.opt.name, epoch, iterations)
        else:
            # 重要修正：確保 label 只是純字串 (如 'latest')，剔除可能夾帶的路徑資訊
            clean_label = str(label).split("/")[-1]
            filename = "%s_%s.pth" % (self.opt.name, clean_label)

        model_path = os.path.join(self.model_save_dir, filename)

        # 執行儲存
        torch.save(self.state_dict(), model_path)
        print(f"===> Model saved to: {model_path}")

    def save_eval(self, label=None):
        if label is None:
            label = "eval_latest"

        # 同樣確保 label 不含路徑
        clean_label = str(label).split("/")[-1]
        filename = "%s.pth" % clean_label

        model_path = os.path.join(self.model_save_dir, filename)
        torch.save(self.state_dict_eval(), model_path)
        print(f"===> Eval model saved to: {model_path}")

    def _init_optimizer(self, optimizers):
        self.optimizers = optimizers
        for optimizer in self.optimizers:
            util.set_opt_param(optimizer, "initial_lr", self.opt.lr)
            util.set_opt_param(optimizer, "weight_decay", self.opt.wd)
