import os, yaml
from pathlib import Path
from omegaconf import OmegaConf

def loadConfig(model_name):
    """
    Loads the config yaml file
    """
    config_path = Path(f"{os.path.dirname(os.getcwd())}/mmf/mmf/configs/models/{('_').join(model_name.split('_')[:-1])}/defaults.yaml")
    with open(config_path, 'r') as stream:
        try:
            parsed_yaml=yaml.safe_load(stream)
            print(parsed_yaml)
        except yaml.YAMLError as exc:
            print(exc)

    # Extract model name
    model_name = config_path.as_posix().split("/")[-2]
    config = OmegaConf.create(parsed_yaml['model_config'][model_name])
    return config