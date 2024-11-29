import os
from lightning.pytorch.callbacks import Callback
from omegaconf import OmegaConf
import structlog
from lightning_project_skeleton.logging.utils import rank_zero_log_only

logger = structlog.get_logger(__name__)


class TrainingSetupLoggerCallback(Callback):
    def __init__(
        self,
        resume: bool,
        now: str,
        logdir: str,
        ckptdir: str,
        cfgdir: str,
        config: dict,
        lightning_config: dict,
        profile: bool = False,
    ):
        super().__init__()
        self._resume = resume
        self._now = now
        self._logdir = logdir
        self._ckptdir = ckptdir
        self._cfgdir = cfgdir
        self._config = config
        self._lightning_config = lightning_config
        self._profile = profile

    def on_fit_start(self, trainer, pl_module):
        if trainer.global_rank == 0:
            # Create logdirs and save configs
            os.makedirs(self._logdir, exist_ok=True)
            os.makedirs(self._ckptdir, exist_ok=True)
            os.makedirs(self._cfgdir, exist_ok=True)

            rank_zero_log_only(logger, "Project config")
            rank_zero_log_only(logger, f"\n{OmegaConf.to_yaml(self._config)}")
            OmegaConf.save(
                self._config,
                os.path.join(self._cfgdir, "{}-project.yaml".format(self._now)),
            )

            rank_zero_log_only(logger, "Lightning config")
            rank_zero_log_only(
                logger, f"\n{OmegaConf.to_yaml(self._lightning_config)}"
            )
            OmegaConf.save(
                OmegaConf.create({"lightning": self._lightning_config}),
                os.path.join(
                    self._cfgdir, "{}-lightning.yaml".format(self._now)
                ),
            )

            # Log code
            pl_module.logger.experiment.log_code(
                root="./",
                name='code',
                include_fn=lambda path: path.endswith(".py")
                or path.endswith(".yaml"),
                exclude_fn=lambda path, root: os.path.relpath(
                    path, root
                ).startswith("training_logs/")
                or os.path.relpath(path, root).startswith("lightning_logs/")
                or os.path.relpath(path, root).startswith("output/")
                or os.path.relpath(path, root).startswith("wandb/"),
            )

            # Model profiling
            if self._profile:
                from deepspeed.profiling.flops_profiler.profiler import (
                    FlopsProfiler,
                )

                _ = (
                    trainer.estimated_stepping_batches
                )  # required to load the dataloader
                prof = FlopsProfiler(pl_module.model)
                profile_step = min(10, len(trainer.train_dataloader) - 1)
                for step, batch in enumerate(trainer.train_dataloader):
                    if step == profile_step:
                        prof.start_profile()
                    batch = list(
                        map(
                            lambda x: x.to(pl_module.device),
                            pl_module.get_input(batch),
                        )
                    )
                    pl_module.model(*batch)

                    if step == profile_step:
                        prof.stop_profile()
                        prof.print_model_profile(
                            profile_step=profile_step,
                            output_file=os.path.join(
                                pl_module.logger.save_dir, "model_profile.txt"
                            ),
                        )
                        prof.end_profile()
                        break

        else:
            # ModelCheckpoint callback created log directory --- remove it
            if not self._resume and os.path.exists(self._logdir):
                dst, name = os.path.split(self._logdir)
                dst = os.path.join(dst, "child_runs", name)
                os.makedirs(os.path.split(dst)[0], exist_ok=True)
                try:
                    os.rename(self._logdir, dst)
                except FileNotFoundError:
                    pass
