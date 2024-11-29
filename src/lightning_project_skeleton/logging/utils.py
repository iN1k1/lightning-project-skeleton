from lightning.pytorch.utilities.rank_zero import rank_zero_only
import structlog


@rank_zero_only
def rank_zero_log_only(logger, msg: str):
    logger.info(msg)


def configure_logger():

    structlog.stdlib.recreate_defaults()
    # Default ConsoleRenderer with columns:
    # timestamp, level, event, logger, logger_name
    default_cr = structlog.stdlib._config._BUILTIN_DEFAULT_PROCESSORS[-1]
    event_col = default_cr._columns[2]
    default_cr._columns[2] = default_cr._columns[3]
    default_cr._columns[3] = event_col

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.add_log_level,
            structlog.stdlib.add_logger_name,
            structlog.processors.StackInfoRenderer(),
            structlog.dev.set_exc_info,
            structlog.processors.TimeStamper(
                fmt="%Y-%m-%d %H:%M:%S", utc=False
            ),
            default_cr,
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.BoundLogger,
        cache_logger_on_first_use=True,
    )


if __name__ == "__main__":
    configure_logger()
    logger = structlog.getLogger(__name__)
    logger.info('info')
    logger.warning('warning')
    logger.error('error')
