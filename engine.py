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

    def train(
        self,
        train_loader,
        custom_save_path=None,
        custom_save_freq=None,
        batch_size=None,
        **kwargs,
    ):
        print("\nEpoch: %d" % self.epoch)
        avg_meters = util.AverageMeters()
        model = self.model
        for i, data in enumerate(train_loader):
            model.set_input(data, mode="train")
            model.optimize_parameters(**kwargs)
            errors = model.get_current_errors()
            avg_meters.update(errors)
            util.progress_bar(i, len(train_loader), str(avg_meters))

            if (
                custom_save_freq is not None
                and (self.iterations + 1) % custom_save_freq == 0
            ):
                trained_images_count = (self.iterations + 1) * batch_size
                save_filename = os.path.join(
                    custom_save_path, f"DSRNet_{trained_images_count}"
                )
                model.save(save_filename)
                print(
                    f">> [CUSTOM SAVE] Iteration {self.iterations + 1}: 已儲存至: {save_filename}"
                )
            self.iterations += 1

        self.epoch += 1
        model.save(label="latest")

    def eval(self, val_loader, dataset_name, savedir="./tmp", suffix=None, **kwargs):
        if savedir is not None:
            os.makedirs(savedir, exist_ok=True)
            self.f = open(os.path.join(savedir, f"metrics_{dataset_name}.txt"), "w+")

        avg_meters = util.AverageMeters()
        model = self.model

        # 用於計算平均 Run time
        total_time = 0

        with torch.no_grad():
            for i, data in enumerate(val_loader):
                # 僅針對推理過程計時
                torch.cuda.synchronize()
                start_time = time.time()

                index = model.eval(data, savedir=savedir, **kwargs)

                torch.cuda.synchronize()
                total_time += time.time() - start_time

                if savedir is not None:
                    self.f.write(
                        f"{data['fn'][0]} PSNR:{index['PSNR']:.4f} SSIM:{index['SSIM']:.4f}\n"
                    )

                avg_meters.update(index)
                util.progress_bar(i, len(val_loader), str(avg_meters))

        avg_runtime = total_time / len(val_loader)
        print(f"\n平均推理時間 (Run time): {avg_runtime:.4f}s")

        if savedir is not None:
            self.f.write(f"\nAverage Metrics: {str(avg_meters)}\n")
            self.f.write(f"Average Runtime: {avg_runtime:.4f}s\n")
            self.f.close()

        return avg_meters

    def test(self, test_loader, savedir=None, **kwargs):
        model = self.model
        with torch.no_grad():
            for i, data in enumerate(test_loader):
                model.test(data, savedir=savedir, **kwargs)
                util.progress_bar(i, len(test_loader))

    @property
    def iterations(self):
        return self.model.iterations

    @iterations.setter
    def iterations(self, i):
        self.model.iterations = i

    @property
    def epoch(self):
        return self.model.epoch

    @epoch.setter
    def epoch(self, e):
        self.model.epoch = e
