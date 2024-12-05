from typing import Union, List, Callable
from lightning_project_skeleton.build.from_config import instantiate_from_config
from lightning import pytorch as pl
import torch
from torch import nn
from torch.nn import init
from torch.nn.modules.batchnorm import _BatchNorm
from lightning.pytorch.utilities import grad_norm
from timm.optim import create_optimizer
from argparse import Namespace
import structlog


_logger = structlog.getLogger(__name__)


class LightningBaseModel(pl.LightningModule):
    def __init__(self, optimizer:dict, scheduler:dict,
                 ignore_keys=[],
                 data_keys:List[str]=["data"],
                 target_keys:List[str]=["target"]):
        super().__init__()
        self.save_hyperparameters()

        # Parameters that we need to specify in order to initialize our model
        self.scheduler_config = scheduler
        self.optimizer_config = optimizer
        self._ignore_keys = ignore_keys
        self._data_keys = data_keys
        self._target_keys = target_keys
        self.model:nn.Module = None


    @torch.no_grad()
    def _default_init_weights(self, module_list:Union[List, nn.Module], scale=1, bias_fill=0, **kwargs):
        """Initialize network weights.

        Args:
            module_list (list[nn.Module] | nn.Module): Modules to be initialized.
            scale (float): Scale initialized weights, especially for residual
                blocks. Default: 1.
            bias_fill (float): The value to fill bias. Default: 0
            kwargs (dict): Other arguments for initialization function.
        """
        if not isinstance(module_list, list):
            module_list = [module_list]

        for module in module_list:
            for m in module.modules():
                if isinstance(m, nn.Conv2d):
                    init.kaiming_normal_(m.weight, **kwargs)
                    m.weight.data *= scale
                    if m.bias is not None:
                        m.bias.data.fill_(bias_fill)
                elif isinstance(m, nn.Linear):
                    init.kaiming_normal_(m.weight, **kwargs)
                    m.weight.data *= scale
                    if m.bias is not None:
                        m.bias.data.fill_(bias_fill)
                elif isinstance(m, _BatchNorm):
                    init.constant_(m.weight, 1)
                    if m.bias is not None:
                        m.bias.data.fill_(bias_fill)
                elif isinstance(m, nn.LayerNorm):
                    init.constant_(m.bias, 0)
                    init.constant_(m.weight, 1.0)

    def init_weights(self, init_fn:Callable=None):
        if init_fn is None:
            init_fn = self._default_init_weights
        self.apply(init_fn)

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")
        
    def configure_optimizers(self):
        opt = create_optimizer(Namespace(**self.optimizer_config), self.model)

        cfg = self.scheduler_config
        cfg['params']['optimizer'] = opt
        scheduler = instantiate_from_config(cfg)

        return [opt], [scheduler]

    def get_input(self, batch):
        return [batch[k] for k in self._data_keys]

    def get_target(self, batch):
        return [batch[k] for k in self._target_keys]

    def on_before_optimizer_step(self, optimizer):
        # Compute the 2-norm for each layer
        # If using mixed precision, the gradients are already unscaled here
        if self.global_step % 100 == 0:
            norms = grad_norm(self.model, norm_type=2)
            self.log_dict(norms)
