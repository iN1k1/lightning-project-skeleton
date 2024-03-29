seed = 0
lightning = dict(
    trainer=dict(
        max_epochs=400,
        check_val_every_n_epoch=50,
        #precision='16-mixed'
    )
)

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
