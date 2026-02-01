import logging
import os
import sys
import yaml
from datetime import datetime

def setup_logging(output_dir: str) -> None:
    """Setup logging configuration"""
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # Remove existing handlers to ensure we control output
    for h in root_logger.handlers[:]:
        root_logger.removeHandler(h)

    formatter = logging.Formatter('[%(asctime)s] %(levelname)s - %(name)s: %(message)s')

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)

    root_logger.addHandler(file_handler)
    root_logger.addHandler(stream_handler)

    logging.info(f"Logging initialized. Saving to {log_file}")


def load_pipeline_config(config_dir: str, pipeline_id: str) -> dict:
    """Load pipeline configuration"""
    config_path = os.path.join(config_dir, f"{pipeline_id}.yaml")
    if not os.path.exists(config_path):
        config_path = os.path.join(config_dir, "pipeline_config.yaml")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    return config


def get_data_dir(dataset_config, dataset_id=None):
    if 'processed_data_root' in dataset_config:
        data_dir = dataset_config['processed_data_root']
    else:
        data_dir = os.path.join(dataset_config.get('data_root', './data'), dataset_id)
    return data_dir
