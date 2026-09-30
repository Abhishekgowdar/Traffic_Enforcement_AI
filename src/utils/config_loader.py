"""
Configuration Loader

Loads and manages configuration from YAML files.
Supports both PyYAML and ruamel.yaml for Python 3.14 compatibility.
"""

from pathlib import Path
from typing import Dict, Any
import logging

# Try PyYAML first, fall back to ruamel.yaml
try:
    import yaml
    YAML_LOADER = 'pyyaml'
except ImportError:
    try:
        from ruamel.yaml import YAML
        yaml = YAML()
        YAML_LOADER = 'ruamel'
    except ImportError:
        raise ImportError("Neither PyYAML nor ruamel.yaml is installed. Install one: pip install ruamel.yaml")

logger = logging.getLogger(__name__)
logger.info(f"Using YAML loader: {YAML_LOADER}")


class ConfigLoader:
    """Centralized configuration management"""

    def __init__(self, config_dir: str = "configs"):
        self.config_dir = Path(config_dir)
        self._model_config = None
        self._app_config = None

    @property
    def model_config(self) -> Dict[str, Any]:
        """Load model configuration"""
        if self._model_config is None:
            config_path = self.config_dir / "model_config.yaml"
            self._model_config = self._load_yaml(config_path)
        return self._model_config

    @property
    def app_config(self) -> Dict[str, Any]:
        """Load application configuration"""
        if self._app_config is None:
            config_path = self.config_dir / "app_config.yaml"
            self._app_config = self._load_yaml(config_path)
        return self._app_config

    def _load_yaml(self, path: Path) -> Dict[str, Any]:
        """Load YAML file"""
        try:
            with open(path, 'r') as f:
                if YAML_LOADER == 'pyyaml':
                    config = yaml.safe_load(f)
                else:  # ruamel.yaml
                    config = yaml.load(f)
            logger.info(f"Loaded config from {path}")
            return config
        except FileNotFoundError:
            logger.error(f"Config file not found: {path}")
            raise
        except Exception as e:
            logger.error(f"Error parsing YAML file {path}: {e}")
            raise

    def get_active_detection_model(self) -> Dict[str, Any]:
        """Get active detection model configuration"""
        active_name = self.model_config['detection']['active_model']
        model_config = self.model_config['detection']['models'][active_name]
        logger.info(f"Active detection model: {active_name}")
        return model_config

    def get_active_ocr_engines(self) -> list:
        """Get list of active OCR engines"""
        active_engines = self.model_config['ocr']['active_engines']
        logger.info(f"Active OCR engines: {', '.join(active_engines)}")
        return active_engines

    def get_active_tracker(self) -> Dict[str, Any]:
        """Get active tracker configuration"""
        active_name = self.model_config['tracking']['active_tracker']
        tracker_config = self.model_config['tracking']['trackers'][active_name]
        logger.info(f"Active tracker: {active_name}")
        return tracker_config

    def get_hardware_config(self) -> Dict[str, Any]:
        """Get hardware configuration"""
        return self.model_config['hardware']

    def reload(self):
        """Reload all configurations"""
        self._model_config = None
        self._app_config = None
        logger.info("Configuration reloaded")


# Global config instance
config = ConfigLoader()
