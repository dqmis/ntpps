import inspect

from omegaconf import DictConfig, OmegaConf

from src.models.neural_tpp import NeuralTPP
from src.models.ode_jump_function import ODEJumpFunction

MODELS_TYPES = {"diffeq": ODEJumpFunction, "ntpp": NeuralTPP}


def parse_model_arguments(conf: DictConfig):
    supported_params = list(inspect.signature(MODELS_TYPES[conf.model_type]).parameters)
    parsed_conf = OmegaConf.to_container(conf, resolve=True)
    return {k: parsed_conf[k] for k in supported_params if k in parsed_conf}


def load_model(conf: DictConfig, model_path: str = None):
    if conf.model_type not in MODELS_TYPES.keys():
        raise Exception(f"Provided model type {conf.model_type} is not found!")
    parsed_params = parse_model_arguments(conf)
    if model_path is None:
        return MODELS_TYPES[conf.model_type](**parsed_params)
    parsed_params["checkpoint_path"] = model_path
    return MODELS_TYPES[conf.model_type].load_from_checkpoint(**parsed_params)
