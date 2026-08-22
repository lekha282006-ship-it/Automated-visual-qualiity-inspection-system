import logging
import os
from logging.handlers import RotatingFileHandler


def setup_logging(log_file: str = "logs/inspection.log", level: int = logging.INFO):
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logger = logging.getLogger()
    if logger.handlers:
        return logger
    logger.setLevel(level)

    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    fh = RotatingFileHandler(log_file, maxBytes=5 * 1024 * 1024, backupCount=5)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


def get_logger(name: str = __name__):
    if not logging.getLogger().handlers:
        setup_logging()
    return logging.getLogger(name)
