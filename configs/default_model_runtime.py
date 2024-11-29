lightning = dict(
    trainer=dict(
        max_epochs=100,
        check_val_every_n_epoch=10,
        gradient_clip_val=1.0,
        precision='bf16-mixed',
    ),
    logging=dict(
        project_name='dummy_example',
    ),
)

model = dict()
