_base_ = ['../default_model_runtime.py']

epochs = 100
warmup_epochs = 25

model = dict(
    target='lightning_project_skeleton.models.base_task.BaseTaskModel',
    params=dict(
        num_classes=2,

        # Data and target keys
        data_keys=['data'],
        target_keys=['target'],

        # Dummy model
        modelconfig=dict(
            target='lightning_project_skeleton.models.example.DummyModel',
            params=dict(
                num_classes=2,
            )
        ),

        # Dummy loss
        lossconfig=dict(
            target='lightning_project_skeleton.loss.example.DummyLoss'
        ),

        # AdamW
        optimizer=dict(
            base_learning_rate=1e-4,
            opt='adamw',
            weight_decay=0.05,
            opt_eps=1e-7,
            momentum=0.9
        ),

        # Linear warmup and cosine annealing
        scheduler=dict(
            target='lightning_project_skeleton.optim.schedulers.LRScheduler',
            params=dict(
                milestones=[warmup_epochs],
                schedulers=[
                    dict(
                        target='torch.optim.lr_scheduler.LinearLR',
                        params=dict(
                            start_factor=0.01,
                            end_factor=1.0,
                            total_iters=warmup_epochs
                        )
                    ),
                    dict(
                        target='torch.optim.lr_scheduler.CosineAnnealingLR',
                        params=dict(
                            T_max=epochs - warmup_epochs,
                            eta_min=1e-5
                        )
                    )
                ]
            )
        )
    )
)
