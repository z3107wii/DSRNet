%%writefile /content/DSRNet/engine.py
import torch
import util.util as util
from models import make_model
import time
import os
import sys
from os.path import join
from util.visualizer import Visualizer

class Engine(object):
    def __init__(self, opt):
        self.opt = opt
        self.writer = None
        self.visualizer = None
        self.model = None
        self.best_val_loss = 1e6
        self.__setup()

    def __setup(self):
        self.basedir = join("checkpoints", self.opt.name)
        os.makedirs(self.basedir, exist_ok=True)
        opt = self.opt
        self.model = make_model(self.opt.model)()
        self.model.initialize(opt)
        if not opt.no_log:
            self.writer = util.get_summary_writer(os.path.join(self.basedir, "logs"))
            self.visualizer = Visualizer(opt)

    def train(self, train_loader, **kwargs):
        # 保持原始 train 邏輯...
        pass

    def eval(self, val_loader, dataset_name, savedir="./tmp", suffix=None, **kwargs):
        """
        核心修正：明確宣告 suffix 參數並向下傳遞
        """
        if savedir is not None:
            os.makedirs(savedir, exist_ok=True)
        
        avg_meters = util.AverageMeters()
        model = self.model
        total_runtime = 0

        with torch.no_grad():
            for i, data in enumerate(val_loader):
                # 精確計時
                torch.cuda.synchronize()
                start_time = time.time()
                
                # 這裡必須明確傳入 suffix=suffix
                index = model.eval(data, savedir=savedir, suffix=suffix, **kwargs)
                
                torch.cuda.synchronize()
                total_runtime += (time.time() - start_time)
                
                avg_meters.update(index)
                util.progress_bar(i, len(val_loader), str(avg_meters))

        # 存入平均推論時間
        avg_meters.dic['Runtime'] = total_runtime / len(val_loader)
        return avg_meters

    def test(self, test_loader, savedir=None, **kwargs):
        model = self.model
        with torch.no_grad():
            for i, data in enumerate(test_loader):
                model.test(data, savedir=savedir, **kwargs)
                util.progress_bar(i, len(test_loader))

    @property
    def iterations(self): return self.model.iterations
    @iterations.setter
    def iterations(self, i): self.model.iterations = i
    @property
    def epoch(self): return self.model.epoch
    @epoch.setter
    def epoch(self, e): self.model.epoch = e