import psutil
import random
import numpy as np
import torch


def log_memory_usage(logger):
    process = psutil.Process()
    mem_info = process.memory_info()
    logger.info(f"Memory Usage: RSS={mem_info.rss / (1024 * 1024):.2f} MB")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
