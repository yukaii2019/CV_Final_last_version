import logging
import os
from arguments import args

def get_logger():

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
    

    file_handler = logging.FileHandler(os.path.join(args.log_root, 'logs.txt'))
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter('%(asctime)s %(message)s'))

    logger = logging.getLogger('logger')
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    logger.setLevel(logging.DEBUG)

    return logger

logger = get_logger()