_base_ = ['../default_data_runtime.py']

data = dict(
    target='lightning_project_skeleton.data.wrapper.DataModuleFromConfig',
    params=dict(
        train_batch_size=128,
        val_batch_size=128,
        num_workers=4,
        train=dict(
            target='lightning_project_skeleton.data.example.ExampleDataset',
            params=dict(
                size = (512,1),
                num_samples = 1000
            )
        ),
        validation=dict(
            target='lightning_project_skeleton.data.example.ExampleDataset',
            params=dict(
                size = (512,1),
                num_samples = 1000
            )
        )
    )
)
