import yaml
import json
import pandas as pd



def read_config_yaml(config_path):
    with open(config_path, "r") as yaml_file:
        config = yaml.load(yaml_file, Loader=yaml.FullLoader)
    return config

def read_config_json(config_path):
    with open(config_path, "r") as json_file:
        config = json.loads(json_file.read())
    return config

def read_txt_dataframe(txtPath: str, sep: str=' ') -> pd.DataFrame:
    df = pd.read_csv(txtPath, sep=sep, index_col=False)
    #df = df.reset_index()
    df.columns = ['image1', 'image2']
    return df  

def config_flatten(config, fconfig):
    for key in config:
        if isinstance(config[key], dict):
            fconfig = config_flatten(config[key], fconfig)
        else:
            fconfig[key] = config[key]
    return fconfig

def config_log(config, keys):
    log = {}
    for key in keys:
        log[key] = config[key]
    return log