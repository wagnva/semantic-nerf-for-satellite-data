Setup Training & Evaluation - Semantic NeRF for Satellite Data
=== 

Create a new `conda` environment:

    conda create -n training python=3.12
    conda activate training

### Cuda

This project was tested using `cuda=12.1.0`

    conda install cuda -c nvidia/label/cuda-12.1.0 

### PyTorch

Install the relevant `PyTorch` version.
This Project was tested using `pytorch=2.4.0`, `torchvision=0.19.0` and `lightning=2.4.0`

    # take care to use the pytorch version to match your cuda version
    pip install torch==2.4.0+cu121 torchvision==0.19.0+cu121 --index-url https://download.pytorch.org/whl/cu121
    pip install lightning==2.4.0


### Pip Dependencies

    pip install kornia==0.5.3 numpy matplotlib pyntcloud toml opencv-python 
    pip install rasterio tensorboard pyproj utm rpcm pymap3d
    pip install plyflatten numba pydantic shapely fire black gpustat scikit-learn pycocotools
    pip3 install -U xformers --index-url https://download.pytorch.org/whl/cu121

### Gdal

Gdal requires both a conda library and a corresponding `pip` python package.
Make sure the versions between them match. 
The latest tested versions are `gdal==3.6.2` and `pygdal==3.6.2.11`.

    conda install gdal
    # check version
    conda list | grep gdal
    pip install pygdal==<version>


### Possible Problems

- `srtm4`:

    **Error**: `Failed building wheel for srtm4`: `fatal error: tiffio.h: No such file or directory`

    **Solution**: `sudo apt-get install libtiff-dev`