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

        """Model"""
        self.model = make_model(self.opt.model)()  # models.__dict__[self.opt.model]()
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
        opt = self.opt
        epoch = self.epoch

        epoch_start_time = time.time()
        for i, data in enumerate(train_loader):
            iter_start_time = time.time()
            iterations = self.iterations

            model.set_input(data, mode="train")
            model.optimize_parameters(**kwargs)

            errors = model.get_current_errors()
            avg_meters.update(errors)
            util.progress_bar(i, len(train_loader), str(avg_meters))

            if not opt.no_log:
                util.write_loss(self.writer, "train", avg_meters, iterations)

                if iterations % opt.display_freq == 0 and opt.display_id != 0:
                    save_result = iterations % opt.update_html_freq == 0
                    self.visualizer.display_current_results(
                        model.get_current_visuals(), epoch, save_result
                    )

            # --- 客製化儲存邏輯 (每 1500 張圖片儲存一次) ---
            if (
                custom_save_freq is not None
                and (self.iterations + 1) % custom_save_freq == 0
            ):
                trained_images_count = (self.iterations + 1) * (
                    batch_size if batch_size else 1
                )
                save_filename = os.path.join(
                    custom_save_path, f"DSRNet_{trained_images_count}"
                )
                model.save(save_filename)
                print(
                    f">> [CUSTOM SAVE] Iteration {self.iterations + 1}: 已儲存模型至: {save_filename}"
                )

            self.iterations += 1

        self.epoch += 1

        if not self.opt.no_log:
            model.save(label="latest")
            print("Time Taken: %d sec" % (time.time() - epoch_start_time))

        try:
            train_loader.reset()
        except:
            pass

    def eval(
        self,
        val_loader,
        dataset_name,
        savedir="./tmp",
        suffix=None,
        max_save_size=None,
        **kwargs,
    ):
        """
        修改後的 eval 函數：
        1. 支援精確 Runtime 計時 (使用 synchronize)
        2. 明確傳遞 suffix 參數避免路徑拼接錯誤
        """
        if savedir is not None:
            os.makedirs(savedir, exist_ok=True)
            self.f = open(os.path.join(savedir, f"metrics_{dataset_name}.txt"), "w+")
            self.f.write(dataset_name + "\n")

        avg_meters = util.AverageMeters()
        model = self.model
        opt = self.opt
        total_runtime = 0

        with torch.no_grad():
            for i, data in enumerate(val_loader):
                if opt.selected and data["fn"][0].split(".")[0] not in opt.selected:
                    continue

                # --- 推理計時 START ---
                torch.cuda.synchronize()
                start_time = time.time()

                # 核心修正：明確傳遞 suffix 給 model.eval
                if max_save_size is not None and i > max_save_size:
                    index = model.eval(data, savedir=None, suffix=suffix, **kwargs)
                else:
                    index = model.eval(data, savedir=savedir, suffix=suffix, **kwargs)

                torch.cuda.synchronize()
                total_runtime += time.time() - start_time
                # --- 推理計時 END ---

                if savedir is not None:
                    self.f.write(
                        f"{data['fn'][0]} {index['PSNR']:.4f} {index['SSIM']:.4f}\n"
                    )

                avg_meters.update(index)
                util.progress_bar(i, len(val_loader), str(avg_meters))

        # 計算平均 Run time 並存入指標字典中
        avg_meters.dic["Runtime"] = total_runtime / len(val_loader)

        if savedir is not None:
            self.f.write(
                f"\nAverage Runtime per image: {avg_meters.dic['Runtime']:.4f}s\n"
            )
            self.f.close()

        if not opt.no_log:
            util.write_loss(
                self.writer, join("eval", dataset_name), avg_meters, self.epoch
            )

        return avg_meters

    def test(self, test_loader, savedir=None, **kwargs):
        model = self.model
        with torch.no_grad():
            for i, data in enumerate(test_loader):
                model.test(data, savedir=savedir, **kwargs)
                util.progress_bar(i, len(test_loader))

    def save_model(self):
        self.model.save()

    def save_eval(self, label):
        self.model.save_eval(label)

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
