import sklearn.pipeline as skpipe
import imblearn.pipeline as iblearn
from omegaconf import DictConfig
import hydra

def make_pipeline_sklearn(
    steps_config: DictConfig,
    ) -> skpipe.Pipeline:
    steps = []
    
    for step_config in steps_config:
        step_name, step_params = list(step_config.items())[0]
        pipeline_step = (step_name, hydra.utils.instantiate(step_params))
        steps.append(pipeline_step)
        
    return skpipe.Pipeline(steps)

def make_pipeline_imblearn(
    steps_config: DictConfig,
    ) -> iblearn.Pipeline:
    steps = []
    
    for step_config in steps_config:
        step_name, step_params = list(step_config.items())[0]
        pipeline_step = (step_name, hydra.utils.instantiate(step_params))
        steps.append(pipeline_step)
        
    return iblearn.Pipeline(steps)
    
    