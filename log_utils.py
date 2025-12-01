import os
import logging


def setup_logging(logger_name=None):
    """
    Configure logging settings with a unified configuration.
    Creates logs directory if it doesn't exist and sets up logging handlers.
    
    Args:
        logger_name: Name for the logger. If None, returns root logger.
    
    Returns:
        Configured logger instance
    """
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    log_file = os.path.join(log_dir, "main.log")
    
    # Check if the root logger already has handlers to avoid duplicate logging
    root_logger = logging.getLogger()
    if not root_logger.handlers:
        # Configure root logger only if it hasn't been configured
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
    
    # Get or create logger with the specified name
    if logger_name:
        logger = logging.getLogger(logger_name)
    else:
        logger = root_logger
    
    # Ensure the logger level is set
    logger.setLevel(logging.INFO)
    
    return logger