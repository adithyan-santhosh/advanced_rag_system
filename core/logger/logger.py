import logging
import os

LOG_FOLDER = "logs"

os.makedirs(LOG_FOLDER, exist_ok=True)

def setup_logger():

    logger = logging.getLogger("rag_logger")
    logger.setLevel(logging.INFO)
    if not logger.handlers:

        file_handler = logging.FileHandler(
            f"{LOG_FOLDER}/rag.log"
        )
        formatter = logging.Formatter(
            "%(asctime)s - %(message)s"
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


logger = setup_logger()