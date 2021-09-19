import logging

from rich.logging import RichHandler

logging.basicConfig(
    format="%(asctime)-15s %(levelname)s - %(funcName)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=[RichHandler()],
)


def get_logger(name) -> logging.Logger:
    return logging.getLogger(name)
