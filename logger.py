import logging
import sys
from logging.handlers import RotatingFileHandler
import os

# Create logs directory
if not os.path.exists('logs'):
    os.makedirs('logs')

def setup_logger(name='BitcoinAI'):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Check if handlers already exist to avoid duplicates
    if logger.hasHandlers():
        return logger

    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # File Handler (Rotating: 5MB max, 2 backups)
    file_handler = RotatingFileHandler('logs/bot.log', maxBytes=5*1024*1024, backupCount=2)
    file_handler.setFormatter(formatter)
    
    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger
