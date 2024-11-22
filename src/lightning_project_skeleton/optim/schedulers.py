import logging
import math
import torch
from typing import List
from torch.optim.lr_scheduler import SequentialLR
from lightning_project_skeleton.build.from_config import instantiate_from_config

_logger = logging.getLogger(__name__)


class LRScheduler(SequentialLR):
    def __init__(
            self,
            optimizer: torch.optim.Optimizer,
            schedulers: List[torch.optim.lr_scheduler._LRScheduler],
            milestones: List[int]) -> None:
        # Build schedulers from configs
        self._schedulers = []
        for cfg in schedulers:
            cfg['params']['optimizer'] = optimizer
            scheduler = instantiate_from_config(cfg)
            self._schedulers.append(scheduler)

        # Sequential scheduler
        super().__init__(optimizer=optimizer,
                         schedulers=self._schedulers,
                         milestones=milestones)


if __name__ == '__main__':
    total_epochs = 400
    warmup_epochs = 25
    test_configs = [
        dict(
            milestones=[warmup_epochs],
            schedulers=[
                dict(
                    target='torch.optim.lr_scheduler.ConstantLR',
                    params=dict(
                        factor=1.0,
                        total_iters=warmup_epochs
                    )
                ),
                dict(
                    target='torch.optim.lr_scheduler.CosineAnnealingLR',
                    params=dict(
                        T_max=total_epochs - warmup_epochs,
                        eta_min=1e-5
                    )
                )
            ]
        ),
        dict(
            milestones=[warmup_epochs],
            schedulers=[
                dict(
                    target='torch.optim.lr_scheduler.LinearLR',
                    params=dict(
                        start_factor=0.0001,
                        end_factor=1.0,
                        total_iters=warmup_epochs
                    )
                ),
                dict(
                    target='torch.optim.lr_scheduler.CosineAnnealingLR',
                    params=dict(
                        T_max=total_epochs - warmup_epochs,
                        eta_min=1e-5
                    )
                )
            ]
        ),

    ]

    for cfg in test_configs:

        # Instantiate the scheduler
        optimizer = torch.optim.AdamW(params=[torch.nn.Parameter(torch.randn(10, 10))])
        scheduler = LRScheduler(optimizer=optimizer, **cfg)

        # Training loop
        lrs = []
        for epoch in range(total_epochs):
            # Record the current learning rate
            lr = optimizer.param_groups[0]['lr']
            lrs.append(lr)

            # Simulate training step
            optimizer.step()

            # Step the scheduler
            scheduler.step()

        # Plotting the learning rate schedule
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 5))
        plt.plot(range(total_epochs), lrs, label='Learning Rate')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.title('Sequential LR Scheduler Check')
        plt.grid(True)
        plt.legend()
        plt.show()
