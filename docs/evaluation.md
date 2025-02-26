Evaluation - Semantic NeRF for Satellite Data
=== 

In this section we describe how you can evaluate the performance of your trained semantic nerf for satellite data.
We include two evaluation scripts, for color (+geometry) and semantic. 
Additionally, we provide a script to extract a series of visualizations. 
Lastly, we also show how you can extract 3D pointclouds of your learned scene.

> [!IMPORTANT]  
> Each script can be applied to a single training directory or a parent directory containing multiple trained models.
> We also provide a handy script to run all evaluation scripts inside a single tmux session

## Evaluation Scripts

### Color & Geometry

This script evaluates the quality of the learned color representation using the two metrics _PSNR_ and _SSIM_. 
Additionally, it computes a *Mean Altitude Error (MAE)*
based on Ground-Truth Lidar Data.

    python -m eval.eval_nerf <input_dp> <output_dp> <split='test'> <epoch='last'> <device=0> <device_req_free=True>

### Semantic

This script evaluates the quality of the learned semantic rendering by computing a semantic accuracy.
It additionally measures the uncertainty $\beta$ for transient locations to measure the impact of our proposed car regularization loss.

    python -m eval.eval_semantic <input_dp> <output_dp> <split='test'> <epoch='last'> <device=0> <device_req_free=True>


### Visualizations

We provide a series of visualizations that are used to visualize the state of our model during training.
During training the visualizations are only computed for a single training view and the test views.
To additionally apply the visualizations to all training images, you can use the two following scripts.
The outputs can be found in the `<out_dir>/visualizations/<split>/<viz_type>/`.

    python -m baseline.run_visualizer <input_dp> <split='test'> <epoch='last'> <device=0> <device_req_free=True> <save_png=False>
    python -m semantic.run_visualizer <input_dp> <split='test'> <epoch='last'> <device=0> <device_req_free=True> <sve_png=False>



###  3D Pointclouds

This script extracts `.ply` pointclouds for each training/test view by converting the learned depth and color of each ray into 3D.

    python -m eval.extract_pointcloud <input_dp> <output_dp> <split='test'> <epoch='last'> <device=0> <device_req_free=True> <max_items=100000>


## Default Output Directory

By setting the `SEMANTIC_SATNERF_EVAL_DP` environment variable, 
a default output directory for the evaluation scripts can be set. 
Place following line in your `~/.bashrc` file:

    export SEMANTIC_SATNERF_EVAL_DP="<path>/<to>/<existing>/<dir>"

## TMUX Utility:

The following script runs all evaluations and visualizations scripts inside a handy tmux session.

    python -m eval.eval <input_dp> <tmux_fp="./tmux_baseline"> <env_name=rs> <gpus=None> <tmux=eval_all>

- `input_dp`: Path to the training/experiment folder
- `tmux_fp`: Path to the tmux `.sh`to use. Options are either `tmux_baseline.sh`or `tmux_semantic.sh`
- `env_name`: Name of the *conda* environment
- `gpus`: List of allowed gpus. If not set chooses from any available, free gpus.
- `tmux`: Name of the *tmux* session

> [!IMPORTANT]  
> The scripts expect a certain amount of available GPUs in your system (3 for baseline, 5 for semantic)
> If you have a different setup, you can adapt the `tmux_baseline.sh`/`tmux_semantic.sh` scripts to your system.

## Gather Evaluation

We provide a handy utility to gather the evaluation results of your models.
If applied to a folder containing multiple evaluated models, it allows for an easy comparison between their performance.
The results are stored inside of `<eval_dp/model/gather_eval/results.txt`.

    python -m eval.gather_eval <eval_dp/model>

> [!IMPORTANT]  
> This script is experimental and probably needs to be adapted to your specific use case / evaluation goal.