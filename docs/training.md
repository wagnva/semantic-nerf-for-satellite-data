Evaluation - Semantic NeRF for Satellite Data
=== 


## Single Training

You can start a single model training using the `start_training.sh` script.
By default, it refers to a `configs/run/default.toml` configuration file. If it does not exist, 
a template one will be created for your customisation.

    sh start_training.sh

> [!IMPORTANT]  
> You can switch between our proposed semantic nerf mode (rs_semantic) or our provided baseline methods (nerf, snerf, satnerf) by switching the name of the
> `pipeline_config` value in the `start_training.sh` script.

### Pipeline Configuration

You can find the configuration files for all provided pipelines inside the 
`configs/pipelines/` directory.

## Multiple Trainings using TMUX 

You can additionally start a series of model trainings inside of a tmux session.
If you have multiple (free) GPUs in your system, they will be run in parallel.
An experiment is defined using a `.toml` file, which allows the user to define multiple training runs overriding 
configuration values of a reference `pipeline.toml` configuration file.
See `confdigs/experiments/satnerf.toml` for reference on how you can use this system.

    python -m run.automated_training <configs/experiments/___.toml>

## Tensorboard

We log the state of the NeRF during training using tensorboard.

    tensorboard --logdir=<model_dir> --port=6006