from typing import List
import torch
import torchmetrics
from torchmetrics import MetricCollection
from torchmetrics.classification import MulticlassAccuracy, MulticlassPrecision, MulticlassRecall
from lightning_project_skeleton.build.from_config import instantiate_from_config
from lightning_project_skeleton.models.base import LightningBaseModel


class BaseTaskModel(LightningBaseModel):
    def __init__(self,
                 modelconfig: dict,
                 lossconfig: dict,
                 optimizer: dict,
                 scheduler: dict,
                 num_classes:int,
                 ignore_keys=[],
                 ckpt_path=None,
                 data_keys: List[str] = ["data"],
                 target_keys: List[str] = ["target"]
                 ):
        super().__init__(optimizer=optimizer,
                         scheduler=scheduler,
                         data_keys=data_keys,
                         target_keys=target_keys,
                         ignore_keys=ignore_keys)

        # Instantiate model
        self.model = instantiate_from_config(modelconfig)

        # Loss
        self.loss = instantiate_from_config(lossconfig)

        # Load checkpoint
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

        self._num_classes = num_classes

        # Train and validation metrics setup
        self.train_metrics = self._init_metrics(['global'], prefix='train')
        self.val_metrics = self._init_metrics(['global'], prefix='val')

    def _init_metrics(self, keys:List[str], prefix:str = 'train', device='cuda'):
        metrics = MetricCollection([
            MulticlassAccuracy(self._num_classes),
            MulticlassPrecision(self._num_classes),
            MulticlassRecall(self._num_classes)
        ])

        return {k: metrics.clone(prefix=f'{prefix}/{k}/').to(device) for k in keys}

    def _forward(self, batch):
        x = self.get_input(batch)
        target = batch['target']
        pred = self.model(*x)
        loss = self.loss(pred, target)
        return pred, target, loss

    def training_step(self, batch, batch_idx):
        pred, target, loss = self._forward(batch)
        bs = target.shape[0]

        m = self.train_metrics['global'](pred, target)
        self.log_dict(m, prog_bar=True, logger=True, on_step=True, on_epoch=True, batch_size=bs, sync_dist=True)
        self.log("train/loss", loss.item(), prog_bar=True, logger=True, on_step=True, on_epoch=True, batch_size=bs,
                 sync_dist=True)
        return loss

    @torch.inference_mode()
    def validation_step(self, batch, batch_idx):
        pred, target, loss = self._forward(batch)
        bs = target.shape[0]

        m = self.val_metrics['global'](pred, target)
        self.log_dict(m, prog_bar=True, logger=True, on_step=False, on_epoch=True, batch_size=bs, sync_dist=True)
        return pred


    @torch.inference_mode()
    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)