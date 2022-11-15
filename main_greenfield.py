from eoles.utils import get_config
from eoles.model_heat_coupling import ModelEOLES
import logging


if __name__ == '__main__':
    config = get_config()

    LOG_FORMATTER = '%(asctime)s : %(name)s  : %(funcName)s : %(levelname)s : %(message)s'
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    # consoler handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(LOG_FORMATTER))
    logger.addHandler(console_handler)
