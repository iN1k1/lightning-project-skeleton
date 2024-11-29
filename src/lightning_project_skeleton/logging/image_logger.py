import numpy as np
import torch
import einops
import os

import wandb
from PIL import Image
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.utilities.rank_zero import rank_zero_only
import torchvision
import structlog

logger = structlog.get_logger(__name__)


class ImageLoggerCallback(Callback):
    def __init__(
        self,
        train_batch_frequency,
        val_batch_frequency,
        max_images,
        preprocess: callable = None,
        clamp=True,
        increase_log_steps=True,
    ):
        super().__init__()
        self._train_batch_freq = train_batch_frequency
        self._val_batch_freq = val_batch_frequency
        self._max_images = max_images
        self._log_steps = [
            2**n for n in range(int(np.log2(self._train_batch_freq)) + 1)
        ]
        if not increase_log_steps:
            self._log_steps = [self._train_batch_freq]
        self._clamp = clamp
        self._preprocess = preprocess

    @rank_zero_only
    def log_remote(self, pl_module, images, batch_idx, split):
        grids = dict()
        for k in images:
            grid = torchvision.utils.make_grid(
                torch.from_numpy(images[k].transpose((0, 3, 1, 2)))
            )
            grids[f"{split}/{k}"] = grid

        pl_module.logger.experiment.log(
            {
                f"{split}-images": [
                    wandb.Image(v, caption=k) for k, v in grids.items()
                ]
            },
            commit=True,
        )

    @rank_zero_only
    def log_local(
        self, save_dir, split, images, global_step, current_epoch, batch_idx
    ):
        root = os.path.join(save_dir, "images", split)
        for k in images:
            grid = torchvision.utils.make_grid(
                torch.from_numpy(images[k].transpose((0, 3, 1, 2))), nrow=4
            )
            grid = einops.rearrange(grid, 'c h w -> h w c').numpy()
            grid = (grid * 255).astype(np.uint8)
            filename = "gs-{:06}_e-{:06}_b-{:06}_{}.png".format(
                global_step, current_epoch, batch_idx, k
            )
            path = os.path.join(root, filename)
            os.makedirs(os.path.split(path)[0], exist_ok=True)
            Image.fromarray(grid).save(path)

    def log_img(self, pl_module, batch, batch_idx, split="train"):
        if (
            self.check_frequency(
                batch_idx, split
            )  # batch_idx % self.batch_freq == 0
            and hasattr(pl_module, "log_images")
            and callable(pl_module.log_images)
            and self._max_images > 0
        ):
            is_train = pl_module.training
            if is_train:
                pl_module.eval()

            with torch.no_grad():
                images = pl_module.log_images(
                    batch, split=split, pl_module=pl_module
                )

            for k in images:
                N = min(images[k].shape[0], self.max_images)
                images[k] = images[k][:N]
                if isinstance(images[k], torch.Tensor):
                    images[k] = (
                        images[k].detach().cpu().numpy().transpose((0, 2, 3, 1))
                    )

                    if self._unnormalize is not None:
                        images[k] = self._unnormalize(image=images[k])['image']

            if self.clamp:
                images = {
                    k: np.clip(images[k], a_min=0.0, a_max=1.0) for k in images
                }

            self.log_local(
                pl_module.logger.save_dir,
                split,
                images,
                pl_module.global_step,
                pl_module.current_epoch,
                batch_idx,
            )

            self.log_remote(pl_module, images, pl_module.global_step, split)

            if is_train:
                pl_module.train()

    def check_frequency(self, batch_idx, split):
        freq = (
            self.train_batch_freq if split == 'train' else self.val_batch_freq
        )
        if (batch_idx % freq) == 0 or (batch_idx in self.log_steps):
            try:
                self.log_steps.pop(0)
            except IndexError:
                pass
            return True
        return False

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.log_img(pl_module, batch, batch_idx, split="train")

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        self.log_img(pl_module, batch, batch_idx, split="val")
