data = dict(
    target='data.wrapper.DataModuleFromConfig',
    params=dict(
        train_batch_size=32,
        val_batch_size=64,
        num_workers=8,
        train=dict(
            target='XYZ',
            params=dict(
            )
        ),
        validation=dict(
            target='XYZ',
            params=dict(
            )
        )
    )
)
