_base_ = ['../default_data_runtime.py']

data = dict(
    target='lightning_project_skeleton.data.wrapper.DataModuleFromConfig',
    params=dict(
        train_batch_size=128,
        val_batch_size=128,
        num_workers=0,
        train=dict(
            target='lightning_project_skeleton.data.example.ExampleDataset',
            params=dict(
                phase='train',
                transform=dict(
                    target='albumentations.Compose',
                    params=[
                        dict(
                            target='albumentations.RandomCrop',
                            params=dict(height=24, width=24),
                        ),
                        dict(
                            target='albumentations.Resize',
                            params=dict(height=32, width=32),
                        ),
                        dict(
                            target='albumentations.HorizontalFlip',
                            params=dict(p=0.5),
                        ),
                        dict(
                            target='albumentations.Normalize',
                            params=dict(
                                mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)
                            ),
                        ),
                    ],
                ),
            ),
        ),
        validation=dict(
            target='lightning_project_skeleton.data.example.ExampleDataset',
            params=dict(
                phase='val',
                transform=dict(
                    target='albumentations.Compose',
                    params=[
                        dict(
                            target='albumentations.Normalize',
                            params=dict(
                                mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)
                            ),
                        ),
                    ],
                ),
            ),
        ),
    ),
)
