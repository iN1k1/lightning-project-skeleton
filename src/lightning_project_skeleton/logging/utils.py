from lightning.pytorch.utilities.rank_zero import rank_zero_only


@rank_zero_only
def rank_zero_log_only(logger, msg:str):
    logger.info(msg)