import psutil


def log_memory_usage(logger):
    process = psutil.Process()
    mem_info = process.memory_info()
    logger.info(f"Memory Usage: RSS={mem_info.rss / (1024 * 1024):.2f} MB")
