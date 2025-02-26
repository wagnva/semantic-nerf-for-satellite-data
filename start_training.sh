#!/bin/bash
run_config=default.toml
pipeline_config=rs_semantic.toml  # nerf | snerf | satnerf | rs_semantic

python -m run.training start_training configs/run/$run_config configs/pipelines/$pipeline_config

