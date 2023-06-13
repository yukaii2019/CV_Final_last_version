from logger import logger
from arguments import args
from trainer import Trainer


if __name__ == '__main__':

    logger.info(str(args)+'\n')
    trainer = Trainer(args)
    trainer.train()