import os, yaml
from pathlib import Path
from omegaconf import OmegaConf

def loadConfig(experiment_name, model_name, studynumber=None):
    """
    Loads the config yaml file
    """
    if studynumber!= None:
        config_path = Path(f"/work3/{studynumber}/Bachelor/save/models/{experiment_name}/config.yaml")
    else:
        config_path = Path(f"{os.getcwd()}/save/models/{experiment_name}/config.yaml")

    with open(config_path, 'r') as stream:
        try:
            parsed_yaml=yaml.safe_load(stream)
            print(parsed_yaml)
        except yaml.YAMLError as exc:
            print(exc)

    # Extract model name
    #model_name = config_path.as_posix().split("/")[-2]
    config = OmegaConf.create(parsed_yaml['model_config'][model_name])
    return config

