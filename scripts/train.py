import argparse, os, sys, datetime, glob, importlib
import signal
import copy
import einops
from omegaconf import OmegaConf
import numpy as np
from PIL import Image
import torch
import torchvision
from lightning.pytorch.utilities.rank_zero import rank_zero_only
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, Callback, LearningRateMonitor
from lightning.pytorch.cli import LightningCLI, LightningArgumentParser
from build.from_config import instantiate_from_config ,load_config_from_py_file
import wandb
from lightning.pytorch.loggers import WandbLogger
import albumentations as A
from lightning_project_skeleton.logging.utils import rank_zero_log_only
logger = logging.getLogger(__name__)


def setup_logging():
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        handlers=[
                            logging.FileHandler("debug.log"),
                            logging.StreamHandler()
                        ])


def get_parser(**parser_kwargs):
    parser = LightningArgumentParser(**parser_kwargs, add_help=False, parse_as_dict=True)
    parser.add_lightning_class_args(L.Trainer, nested_key='trainer')

    parser.add_argument(
        "-n",
        "--name",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="postfix for logdir",
    )
    parser.add_argument(
        "-r",
        "--resume",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="resume from logdir or checkpoint in logdir",
    )
    parser.add_argument(
        "-cfg",
        "--config",
        type=str,
        help="paths to config ",
        default='./configs/idenet_x4_32_setting1.py',
        # default='./configs/idenet_x4_32_real.py',
    )
    parser.add_argument(
        "-p",
        "--project",
        help="name of new or path to existing project",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=23,
        help="seed for seed_everything",
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
        help="Logging path",
    )
    return parser


def nondefault_trainer_args(opt):
    parser = LightningArgumentParser(add_help=False, parse_as_dict=False)
    parser.add_lightning_class_args(L.Trainer, None)
    args = parser.parse_args([])
    return sorted(k for k in vars(args) if getattr(opt, k) != getattr(args, k))

def parse_args():
    parser = get_parser()
    # opt, unknown = parser.parse_known_args()
    opt = parser.parse_args()
    return argparse.Namespace(**opt)

class SetupCallback(Callback):
    def __init__(self, resume, now, logdir, ckptdir, cfgdir, config, lightning_config):
        super().__init__()
        self.resume = resume
        self.now = now
        self.logdir = logdir
        self.ckptdir = ckptdir
        self.cfgdir = cfgdir
        self.config = config
        self.lightning_config = lightning_config

    def on_fit_start(self, trainer, pl_module):
        if trainer.global_rank == 0:
            # Create logdirs and save configs
            os.makedirs(self.logdir, exist_ok=True)
            os.makedirs(self.ckptdir, exist_ok=True)
            os.makedirs(self.cfgdir, exist_ok=True)

            rank_zero_log_only(logger, "Project config")
            rank_zero_log_only(logger, OmegaConf.to_yaml(self.config))
            OmegaConf.save(self.config,
                           os.path.join(self.cfgdir, "{}-project.yaml".format(self.now)))

            rank_zero_log_only(logger, "Lightning config")
            rank_zero_log_only(logger, OmegaConf.to_yaml(self.lightning_config))
            OmegaConf.save(OmegaConf.create({"lightning": self.lightning_config}),
                           os.path.join(self.cfgdir, "{}-lightning.yaml".format(self.now)))

            # Log code
            pl_module.logger.experiment.log_code(
                root="./",
                name='code',
                include_fn=lambda path: path.endswith(".py") or path.endswith(".yaml"),
                exclude_fn=lambda path, root: os.path.relpath(path, root).startswith("training_logs/") or
                                              os.path.relpath(path, root).startswith("lightning_logs/") or
                                              os.path.relpath(path, root).startswith("output/") or
                                              os.path.relpath(path, root).startswith("wandb/")
            )

            # Model profiling
            # from deepspeed.profiling.flops_profiler.profiler import FlopsProfiler
            # _ = trainer.estimated_stepping_batches # required to load the dataloader
            # prof = FlopsProfiler(pl_module.model)
            # profile_step = 10
            # for step, batch in enumerate(trainer.train_dataloader):
            #     if step == profile_step:
            #         prof.start_profile()
            #     batch = list(map(lambda x:x.to(pl_module.device), pl_module.get_input(batch)))
            #     pl_module.model(*batch)
            #
            #     if step == profile_step:
            #         prof.stop_profile()
            #         # flops = prof.get_total_flops(as_string=True)
            #         # params = prof.get_total_params(as_string=True)
            #         prof.print_model_profile(profile_step=profile_step, output_file=os.path.join(pl_module.logger.save_dir, "model_profile.txt"))
            #         prof.print_model_aggregated_profile()
            #         # pl_module.logger.experiment.log_text(
            #         prof.end_profile()
            #         break

        else:
            # ModelCheckpoint callback created log directory --- remove it
            if not self.resume and os.path.exists(self.logdir):
                dst, name = os.path.split(self.logdir)
                dst = os.path.join(dst, "child_runs", name)
                os.makedirs(os.path.split(dst)[0], exist_ok=True)
                try:
                    os.rename(self.logdir, dst)
                except FileNotFoundError:
                    pass


class ImageLogger(Callback):
    def __init__(self, unnormalize:dict, train_batch_frequency, val_batch_frequency,
                 max_images, clamp=True, increase_log_steps=True):
        super().__init__()
        self.train_batch_freq = train_batch_frequency
        self.val_batch_freq = val_batch_frequency
        self.max_images = max_images
        self.logger_log_images = {
            WandbLogger: self._wandb,
            # pl.loggers.TestTubeLogger: self._testtube,
        }
        self.log_steps = [2 ** n for n in range(int(np.log2(self.train_batch_freq)) + 1)]
        if not increase_log_steps:
            self.log_steps = [self.train_batch_freq]
        self.clamp = clamp
        self._unnormalize = A.Compose([A.Normalize(mean=[-m/s for m, s in zip(unnormalize['mean'],unnormalize['std'])],
                                                  std=[1.0 / s for s in unnormalize['std']],
                                                   max_pixel_value=1.0)]) if unnormalize is not None else None

    @rank_zero_only
    def _wandb(self, pl_module, images, batch_idx, split):
        # raise ValueError("No way wandb")
        grids = dict()
        for k in images:
            grid = torchvision.utils.make_grid(torch.from_numpy(images[k].transpose((0,3,1,2))))
            grids[f"{split}/{k}"] = grid
            # pl_module.logger.log_image(f"dassa", images=[np.zeros((256, 256, 3))])
            # pl_module.logger.experiment.log({f'{k}':wandb.Image(grid)}, commit=True)
            # pl_module.logger.log_image(key=f'{k}', images=[grid])
            # pl_module.logger.experiment.log(grids)

        pl_module.logger.experiment.log({"Val Images": [wandb.Image(v, caption=k)
                                                         for k,v in grids.items()]},
                                        commit=True)

    @rank_zero_only
    def log_local(self, save_dir, split, images,
                  global_step, current_epoch, batch_idx):
        root = os.path.join(save_dir, "images", split)
        for k in images:
            grid = torchvision.utils.make_grid(torch.from_numpy(images[k].transpose((0,3,1,2))), nrow=4)
            grid = einops.rearrange(grid, 'c h w -> h w c').numpy()
            grid = (grid*255).astype(np.uint8)
            filename = "gs-{:06}_e-{:06}_b-{:06}_{}.png".format(
                global_step,
                current_epoch,
                batch_idx,
                k)
            path = os.path.join(root, filename)
            os.makedirs(os.path.split(path)[0], exist_ok=True)
            Image.fromarray(grid).save(path)

    def log_img(self, pl_module, batch, batch_idx, split="train"):
        if (self.check_frequency(batch_idx, split) and  # batch_idx % self.batch_freq == 0
                hasattr(pl_module, "log_images") and
                callable(pl_module.log_images) and
                self.max_images > 0):
            logger = type(pl_module.logger)

            is_train = pl_module.training
            if is_train:
                pl_module.eval()

            with torch.no_grad():
                images = pl_module.log_images(batch, split=split, pl_module=pl_module)

            for k in images:
                N = min(images[k].shape[0], self.max_images)
                images[k] = images[k][:N]
                if isinstance(images[k], torch.Tensor):
                    images[k] = images[k].detach().cpu().numpy().transpose((0, 2, 3, 1))

                    if self._unnormalize is not None:
                        images[k] = self._unnormalize(image=images[k])['image']

            if self.clamp:
                images = {k: np.clip(images[k], a_min=0., a_max=1.) for k in images}

            self.log_local(pl_module.logger.save_dir, split, images,
                           pl_module.global_step, pl_module.current_epoch, batch_idx)

            logger_log_images = self.logger_log_images.get(logger, lambda *args, **kwargs: None)
            logger_log_images(pl_module, images, pl_module.global_step, split)

            if is_train:
                pl_module.train()

    def check_frequency(self, batch_idx, split):
        freq = self.train_batch_freq if split == 'train' else self.val_batch_freq
        if (batch_idx % freq) == 0 or (batch_idx in self.log_steps):
            try:
                self.log_steps.pop(0)
            except IndexError:
                pass
            return True
        return False

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.log_img(pl_module, batch, batch_idx, split="train")

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        self.log_img(pl_module, batch, batch_idx, split="val")


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
    setup_logging()
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
        nowname = _tmp[-1]#_tmp.index("training_logs") + 1]
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
    trainer_config = OmegaConf.merge(*[OmegaConf.create(vars(opt)).trainer,
                                       lightning_config.get("trainer", OmegaConf.create())])
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
                "project": lightning_config.get("logging", OmegaConf.create()).get("project_name", "default"),
                "tags": lightning_config.get("logging", OmegaConf.create()).get("tags", None),
                "save_code": True,
                "settings" : {"code_dir":".."}
            }
        }
    }
    default_logger_cfg = default_logger_cfgs["wandb"]
    logger_cfg = OmegaConf.create()
    
    logger_cfg = OmegaConf.merge(default_logger_cfg, logger_cfg)
    trainer_kwargs["logger"] = instantiate_from_config(logger_cfg)

    # add callback which sets up log directory
    image_denormalize = None
    if 'transform' in config.data.params.train.params:
        if p := list(filter(lambda cfg: 'albumentations.Normalize' in cfg.target, config.data.params.train.params.transform)):
            image_denormalize = p[0].params

    default_callbacks_cfg = {
        "setup_callback": {
            "target": "train.SetupCallback",
            "params": {
                "resume": opt.resume,
                "now": now,
                "logdir": logdir,
                "ckptdir": ckptdir,
                "cfgdir": cfgdir,
                "config": copy.deepcopy(config),
                "lightning_config": lightning_config,
            }
        },
        "image_logger": {
            "target": "train.ImageLogger",
            "params": {
                "train_batch_frequency": 200,
                "val_batch_frequency": 10,
                "max_images": 4,
                "unnormalize": image_denormalize,
                "clamp": True
            }
        },
        "learning_rate_logger": {
           "target": "train.LearningRateMonitor",
           "params": {
               "logging_interval": "step",
           }
        },
        "checkpoint_callback": {
                "target": "lightning.pytorch.callbacks.ModelCheckpoint",
                "params": {
                    "dirpath": ckptdir,
                    "filename": "{epoch:04}",
                    "verbose": True,
                    "save_last": True,
                    "monitor": "val/log_metric",
                    "save_top_k": 3,
                    "mode": "max"
                }
        }

    }

    if hasattr(model, "monitor"):
        print(f"Monitoring {model.monitor} as checkpoint metric.")
        default_callbacks_cfg["checkpoint_callback"]["params"]["monitor"] = model.monitor
        default_callbacks_cfg["checkpoint_callback"]["params"]["save_top_k"] = 3

    # data
    data = instantiate_from_config(config.data)
    # NOTE according to https://pytorch-lightning.readthedocs.io/en/latest/datamodules.html
    # calling these ourselves should not be necessary but it is.
    # lightning still takes care of proper multiprocessing though
    data.prepare_data()
    data.setup()

    callbacks_cfg = OmegaConf.create()
    callbacks_cfg = OmegaConf.merge(default_callbacks_cfg, callbacks_cfg)
    trainer_kwargs["callbacks"] = [instantiate_from_config(callbacks_cfg[k]) for k in callbacks_cfg]

    # Setup trainer
    trainer = L.Trainer(**(vars(trainer_opt) | trainer_kwargs))

    # configure learning rate
    bs, base_lr = config.data.params.train_batch_size, config.model.params.optimizer.base_learning_rate
    if not cpu:
        ngpu = int(trainer_config['devices'])
    else:
        ngpu = 1
    accumulate_grad_batches = lightning_config.trainer.accumulate_grad_batches or 1
    rank_zero_log_only(logger, f"accumulate_grad_batches = {accumulate_grad_batches}")
    lightning_config.trainer.accumulate_grad_batches = accumulate_grad_batches
    lr = accumulate_grad_batches * ngpu * bs * base_lr
    rank_zero_log_only(logger, f"Setting learning rate to {lr:.2e} = {accumulate_grad_batches} "
                               f"(accumulate_grad_batches) * {ngpu} (num_gpus) * {bs} (batchsize) * {base_lr:.2e} (base_lr)")

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
        trainer.fit(model, data, ckpt_path=opt.resume_from_checkpoint if opt.resume else None)
    except Exception:
        melk()
        raise