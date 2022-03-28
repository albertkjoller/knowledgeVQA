import os, yaml
from pathlib import Path
from omegaconf import OmegaConf

def loadConfig(model_name, ):
    """
    Loads the config yaml file
    """
    # TODO: remove when not debuggin
    if os.getcwd().split(os.sep)[-1] == 'mmf':
        config_path = Path(f"{os.getcwd()}/save/models/{('_').join(model_name.split('_')[:-1])}/config.yaml")
    else:
        config_path = Path(f"{os.getcwd()}/mmf/save/models/{('_').join(model_name.split('_')[:-1])}/config.yaml")

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

