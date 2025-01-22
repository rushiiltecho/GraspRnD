import yaml

class LoadConfig:
    def __init__(self, config_path):
        """
        Load configuration from a YAML file.
        
        Args:
            config_path (str): Path to the configuration YAML file.
        """
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)

    def get(self, key, default=None):
        """
        Get a configuration value.
        
        Args:
            key (str): Key for the configuration setting.
            default: Default value if the key is not found.
        
        Returns:
            The value associated with the key, or the default value.
        """
        keys = key.split('.')
        value = self.config
        for k in keys:
            if k in value:
                value = value[k]
            else:
                return default
        return value
