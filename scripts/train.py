import argparse
import copy
import datetime
import glob
import os
import signal
import sys

import lightning as L
import structlog
from lightning.pytorch.cli import LightningArgumentParser
from omegaconf import OmegaConf

from lightning_project_skeleton.build.from_config import (
    instantiate_from_config,
    load_config_from_py_file,
)
from lightning_project_skeleton.logging.utils import (
    rank_zero_log_only,
    configure_logger,
)

logger = structlog.getLogger(__name__)


def get_parser(**parser_kwargs):
    parser = LightningArgumentParser(
        **parser_kwargs, add_help=False, parse_as_dict=True
    )
    parser.add_lightning_class_args(L.Trainer, nested_key='trainer')

    parser.add_argument(
        "-cfg",
        "--config",
        type=str,
        required=True,
        help="Paths to *.py config file (required)",
    )
    parser.add_argument(
        "-n",
        "--name",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="Postfix for logdir (default='')",
    )
    parser.add_argument(
        "-r",
        "--resume",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="Resume from _logdir or checkpoint in _logdir (default='')",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=23,
        help="Lightning seed for seed_everything (default=23)",
    )
    parser.add_argument(
        "-f",
        "--postfix",
        type=str,
        default="",
        help="post-postfix for default name",
    )
    parser.add_argument(
        "-ld",
        "--logdir",
        type=str,
        default="./training_logs",
        help="Logging path (default=./training_logs)",
    )
    parser.add_argument(
        "-p",
        "--profile",
        action="store_true",
        help="Perform model profiling",
    )
    return parser


def nondefault_trainer_args(opt):
    parser = LightningArgumentParser(add_help=False, parse_as_dict=False)
    parser.add_lightning_class_args(L.Trainer, None)
    args = parser.parse_args([])
    return sorted(k for k in vars(args) if getattr(opt, k) != getattr(args, k))


def parse_args():
    parser = get_parser()
    opt = parser.parse_args()
    return argparse.Namespace(**opt)


if __name__ == "__main__":
    # custom parser to specify config files, train, test and debug mode,
    # postfix, resume.
    # `--key value` arguments are interpreted as arguments to the trainer.
    # `nested.key=value` arguments are interpreted as config parameters.
    # configs are merged from left-to-right followed by command line parameters.

    # model:
    #   base_learning_rate: float
    #   target: path to lightning module
    #   params:
    #       key: value
    # data:
    #   target: main.DataModuleFromConfig
    #   params:
    #      batch_size: int
    #      wrap: bool
    #      train:
    #          target: path to train dataset
    #          params:
    #              key: value
    #      validation:
    #          target: path to validation dataset
    #          params:
    #              key: value
    #      test:
    #          target: path to test dataset
    #          params:
    #              key: value
    # lightning: (optional, has sane defaults and can be specified on cmdline)
    #   trainer:
    #       additional arguments to trainer
    #   logger:
    #       logger to instantiate
    #   modelcheckpoint:
    #       modelcheckpoint to instantiate
    #   callbacks:
    #       callback1:
    #           target: importpath
    #           params:
    #               key: value
    configure_logger()
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")

    # add cwd for convenience and to make classes in this file available when
    # running as `python main.py`
    # (in particular `main.DataModuleFromConfig`)
    sys.path.append(os.getcwd())

    # Parse args
    opt = parse_args()

    if opt.name and opt.resume:
        raise ValueError(
            "-n/--name and -r/--resume cannot be specified both."
            "If you want to resume training in a new log folder, "
            "use -n/--name in combination with --resume_from_checkpoint"
        )
    if opt.resume:
        if not os.path.exists(opt.resume):
            raise ValueError("Cannot find {}".format(opt.resume))
        if os.path.isfile(opt.resume):
            paths = opt.resume.split("/")
            idx = len(paths) - paths[::-1].index(Path(opt.logdir).stem) + 1
            logdir = "/".join(paths[:idx])
            ckpt = opt.resume
        else:
            assert os.path.isdir(opt.resume), opt.resume
            logdir = opt.resume.rstrip("/")
            ckpt = os.path.join(logdir, "checkpoints", "last.ckpt")

        opt.resume_from_checkpoint = ckpt
        base_configs = sorted(glob.glob(os.path.join(logdir, "configs/*.yaml")))
        # opt.config = base_configs + opt.config
        _tmp = logdir.split("/")
        nowname = _tmp[-1]  # _tmp.index("training_logs") + 1]
    else:
        if opt.name:
            name = "_" + opt.name
        elif opt.config:
            cfg_fname = os.path.split(opt.config)[-1]
            cfg_name = os.path.splitext(cfg_fname)[0]
            name = "_" + cfg_name
        else:
            name = ""
        nowname = now + name + opt.postfix
        logdir = os.path.join(opt.logdir, nowname)

    ckptdir = os.path.join(logdir, "checkpoints")
    cfgdir = os.path.join(logdir, "configs")
    L.seed_everything(opt.seed)

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
    # default to ddp
    trainer_config["strategy"] = "ddp_find_unused_parameters_true"
    if trainer_config['accelerator'] != "gpu":
        del trainer_config["strategy"]
        cpu = True
    else:
        gpuinfo = trainer_config['devices']
        rank_zero_log_only(logger, f"Running on {gpuinfo} GPUs")
        cpu = False
    trainer_opt = argparse.Namespace(**trainer_config)
    lightning_config.trainer = trainer_config

    # model
    model = instantiate_from_config(OmegaConf.to_object(config.model))

    # trainer and callbacks
    trainer_kwargs = dict()

    # default logger configs
    default_logger_cfgs = {
        "wandb": {
            "target": "lightning.pytorch.loggers.WandbLogger",
            "params": {
                "name": nowname,
                "save_dir": logdir,
                "offline": False,
                "id": nowname,
                "project": lightning_config.get(
                    "logging", OmegaConf.create()
                ).get("project_name", "default"),
                "tags": lightning_config.get("logging", OmegaConf.create()).get(
                    "tags", None
                ),
                "save_code": True,
                "settings": {"code_dir": ".."},
            },
        }
    }
    default_logger_cfg = default_logger_cfgs["wandb"]
    logger_cfg = OmegaConf.create()

    logger_cfg = OmegaConf.merge(default_logger_cfg, logger_cfg)
    trainer_kwargs["logger"] = instantiate_from_config(logger_cfg)

    default_callbacks_cfg = {
        "setup_callback": {
            "target": "lightning_project_skeleton.logging.training_setup_logger.TrainingSetupLoggerCallback",
            "params": {
                "resume": opt.resume,
                "now": now,
                "logdir": logdir,
                "ckptdir": ckptdir,
                "cfgdir": cfgdir,
                "config": copy.deepcopy(config),
                "lightning_config": lightning_config,
                "profile": opt.profile,
            },
        },
        # "image_logger": {
        #     "target": "lightning_project_skeleton.logging.image_logger.ImageLoggerCallback",
        #     "params": {
        #         "train_batch_frequency": 10,
        #         "val_batch_frequency": 5,
        #         "max_images": 4,
        #         "clamp": True,
        #     },
        # },
        "learning_rate_logger": {
            "target": "lightning.pytorch.callbacks.LearningRateMonitor",
            "params": {
                "logging_interval": "step",
            },
        },
        "checkpoint_callback": {
            "target": "lightning.pytorch.callbacks.ModelCheckpoint",
            "params": {
                "dirpath": ckptdir,
                "filename": "{epoch:04}",
                "verbose": True,
                "save_last": True,
                "monitor": "val/global/MulticlassAccuracy",
                "save_top_k": 3,
                "mode": "max",
            },
        },
    }

    if hasattr(model, "monitor"):
        print(f"Monitoring {model.monitor} as checkpoint metric.")
        default_callbacks_cfg["checkpoint_callback"]["params"][
            "monitor"
        ] = model.monitor
        default_callbacks_cfg["checkpoint_callback"]["params"]["save_top_k"] = 3

    # Setup data
    data = instantiate_from_config(config.data)

    # Add callbacks
    callbacks_cfg = OmegaConf.create(config.get('callbacks', {}))
    callbacks_cfg = OmegaConf.merge(default_callbacks_cfg, callbacks_cfg)
    trainer_kwargs["callbacks"] = [
        instantiate_from_config(callbacks_cfg[k]) for k in callbacks_cfg
    ]

    # Setup trainer
    trainer = L.Trainer(**(vars(trainer_opt) | trainer_kwargs))

    # configure learning rate
    bs, base_lr = (
        config.data.params.train_batch_size,
        config.model.params.optimizer.base_learning_rate,
    )
    if not cpu:
        ngpu = int(trainer_config['devices'])
    else:
        ngpu = 1
    accumulate_grad_batches = (
        lightning_config.trainer.accumulate_grad_batches or 1
    )
    rank_zero_log_only(
        logger, f"accumulate_grad_batches = {accumulate_grad_batches}"
    )
    lightning_config.trainer.accumulate_grad_batches = accumulate_grad_batches
    lr = accumulate_grad_batches * ngpu * bs * base_lr
    model.optimizer_config['lr'] = lr
    rank_zero_log_only(
        logger,
        f"Setting learning rate to {lr:.2e} = {accumulate_grad_batches} "
        f"(accumulate_grad_batches) * {ngpu} (num_gpus) * {bs} (batchsize) * {base_lr:.2e} (base_lr)",
    )

    # allow checkpointing via USR1
    def melk(*args, **kwargs):
        # run all checkpoint hooks
        if trainer.global_rank == 0:
            rank_zero_log_only(logger, "Summoning checkpoint.")
            ckpt_path = os.path.join(ckptdir, "last.ckpt")
            trainer.save_checkpoint(ckpt_path)

    def divein(*args, **kwargs):
        if trainer.global_rank == 0:
            import pudb

            pudb.set_trace()

    signal.signal(signal.SIGUSR1, melk)
    signal.signal(signal.SIGUSR2, divein)

    # run
    try:
        trainer.fit(
            model,
            data,
            ckpt_path=opt.resume_from_checkpoint if opt.resume else None,
        )
    except Exception:
        melk()
        raise
