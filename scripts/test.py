import argparse, os
from omegaconf import OmegaConf
import lightning as L
from lightning.pytorch.cli import LightningArgumentParser
from lightning_project_skeleton.build.from_config import (
    instantiate_from_config,
    load_config_from_py_file,
)


def get_parser(**parser_kwargs):

    parser = LightningArgumentParser(
        **parser_kwargs, add_help=False, parse_as_dict=True
    )
    parser.add_lightning_class_args(L.Trainer, nested_key='trainer')

    parser.add_argument(
        '-ckpt',
        '--checkpoint',
        required=True,
        type=str,
        help='Model checkpoint path',
    )
    parser.add_argument(
        '-cfg', '--config', required=True, type=str, help="Config path"
    )
    parser.add_argument(
        '-o',
        '--output-dir',
        type=str,
        help="Output directory path (default: ./output)",
        default='./output',
    )

    return parser


def parse_args():
    parser = get_parser()
    opt = parser.parse_args()
    return argparse.Namespace(**opt)


if __name__ == "__main__":

    # Parse args
    opt = parse_args()

    # Make sure output dir exists
    os.makedirs(opt.output_dir, exist_ok=True)

    # Load config
    config = OmegaConf.create(load_config_from_py_file(opt.config))

    # merge trainer cli with config
    lightning_config = config.pop("lightning", OmegaConf.create())
    trainer_config = OmegaConf.merge(
        *[
            OmegaConf.create(vars(opt)).trainer,
            lightning_config.get("trainer", OmegaConf.create()),
        ]
    )
    trainer_opt = argparse.Namespace(**trainer_config)
    lightning_config.trainer = trainer_config

    # Load model from checkpoint
    model = instantiate_from_config(OmegaConf.to_object(config.model))

    # Ensure 32bit precision at validation
    trainer_kwargs = {'precision': '32-true'}

    # TODO: Add custom callbacks if needed
    # default_callbacks_cfg = {
    #     "score_logger_callback": {
    #         "target": "gait.logging.score_logger.ScoreLoggerCallback",
    #         "params": {
    #             "_logdir": opt.output_dir,
    #         }
    #     }
    # }
    # # callbacks_cfg = lightning_config.callbacks or OmegaConf.create()
    # callbacks_cfg = OmegaConf.create()
    # callbacks_cfg = OmegaConf.merge(default_callbacks_cfg, callbacks_cfg)
    # trainer_kwargs["callbacks"] = [instantiate_from_config(callbacks_cfg[k]) for k in callbacks_cfg]

    # Setup trainer object
    trainer = L.Trainer(**(vars(trainer_opt) | trainer_kwargs))

    # Bootstrap data while removing any useless training set to gain time
    if 'train' in config.data.params:
        del config.data.params.train
    data = instantiate_from_config(config.data)
    data.prepare_data()
    data.setup()

    # Run validation using the given checkpoint
    trainer.validate(model, dataloaders=data, ckpt_path=opt.checkpoint)
