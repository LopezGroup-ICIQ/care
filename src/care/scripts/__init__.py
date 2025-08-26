import logging
import warnings
warnings.filterwarnings(
    "ignore",
    message=r".*torch.load.*weights_only=False.*",
    category=FutureWarning,
)
warnings.filterwarnings("ignore", category=FutureWarning, message=".*ExpCellFilter.*")
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=r".*torch\.cuda\.amp\.autocast.*deprecated.*"
)

import dask
from distributed.comm import CommClosedError


def setup_logging(log_file=None):
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.ERROR)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.ERROR)
        # Remove ALL pre-existing handlers
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        root_logger.addHandler(file_handler)

        # Redirect warnings through logging
        logging.captureWarnings(True)
        warnings.filterwarnings("ignore", category=FutureWarning)

        # Clean and redirect specific noisy loggers (like bokeh)
        for name in [
            "acat"
            "distributed",
            "bokeh",
            "tornado",
            "ase",
            "torch",
            "asyncio",
            "torch._dynamo",
            "mace"
        ]:           
            logger = logging.getLogger(name)
            logger.setLevel(logging.ERROR)
            # Remove all their handlers
            for handler in logger.handlers[:]:
                logger.removeHandler(handler)
            logger.addHandler(file_handler)
            logger.propagate = False  # ensure they don't write to parent stdout handlers
        class CommClosedFilter(logging.Filter):
            def filter(self, record):
                return "CommClosedError" not in record.getMessage()

        for name in ["distributed.worker", "distributed.comm.tcp"]:
            logging.getLogger(name).addFilter(CommClosedFilter())
    else:
        logging.basicConfig(level=logging.WARNING)

@dask.delayed
def load_x(x):
    return x

@dask.delayed
def predict(x, dmodel):
    dmodel(x)
    return x