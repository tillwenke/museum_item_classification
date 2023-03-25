import hydra
import omegaconf
from pathlib import Path
from datetime import datetime
from train_utils import *

cfg = None
root_path = None
run_name = None
best_result = None

def init_globals(config, name):
    global cfg
    global root_path
    global run_name
    global best_result
    
    cfg = config
    root_path = Path(os.getcwd())
    run_name = name
    best_result = 0

def make_parameters(hyperparams):
    parameters = {}
    for key, value in hyperparams.items():
        if isinstance(value, list):
            parameters[key] = {'values': value}
        else:
            parameters[key] = value
    return parameters


def run_training():
    global cfg
    global root_path
    global run_name
    global best_result
    
    run = wandb.init()
    params = wandb.config

    path = root_path / cfg['dataset']['data_dir'] / cfg['dataset']['file']
    print(cfg['dataset']['data_dir'])
    if 'column' in cfg['dataset']['data_dir']:
        train, val, test = get_data(path=path)
    elif 'text' in cfg['dataset']['data_dir']:
        train, val, test = get_bow(text_path=path, column_path=root_path / cfg['dataset']['extra_dir'] / cfg['dataset']['extra_file'])
    elif 'embeddings' in cfg['dataset']['data_dir']:
        train, val, test = get_embeddings(path=path)
    else:
        raise ValueError('No valid dataset was found')

    # Creating pipeline
    pipeline = hydra.utils.instantiate(cfg["pipeline"]["init"], _recursive_=False)
    pipeline = pipeline.set_params(**params)

    X = train.drop(columns=['type'])
    y = train['type']

    pipeline.fit(X, y)

    X_val = val.drop(columns=['type'])
    y_val = val['type']

    predictions = pipeline.predict(X_val)

    acc =  accuracy_score(y_val, predictions)
    f1 = f1_score(y_val, predictions, average='macro')
    wandb.log({
      'acc': acc,
      'f1_macro': f1
    })

    if f1 > best_result:
        best_result = f1
        print(pipeline)
        model = pipeline['model']
        print(model)
        file = f"{root_path}/models/{cfg['dataset'].name}/best_model_{run_name}_{f1}.pkl"
        print(file)
        pickle.dump(model, open(file, "wb"))    


@hydra.main(config_path="./conf", config_name="config")
def train(cfg):
    run_name = f"{cfg['pipeline'].name}_{cfg['dataset'].name}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    init_globals(cfg, run_name)

    sweep_configuration = omegaconf.OmegaConf.to_container(cfg['parameters'], resolve=True, throw_on_missing=True)

    # Initialize sweep by passing in config. (Optional) Provide a name of the project.
    project = f"{cfg['pipeline'].name}_{cfg['dataset'].name}"
    sweep_id = wandb.sweep(sweep=sweep_configuration, project=project)

    # Start sweep job.
    wandb.agent(sweep_id, function=run_training, count=5)

if __name__ == "__main__":
    train()