Dataset Preprocessing - Semantic NeRF for Satellite Data
=== 

This section describes the functionality of the ``data_prep.create_dataset`` script. 
This script combines the [DFC2019 - Track-3](https://ieee-dataport.org/open-access/data-fusion-contest-2019-dfc2019) dataset
with our provided manually generated semantic [annotations(dataset.zip),
The script performs cropping to a defined Region-of-Interest, Metadata extraction, 
Bundle Adjustment and Train/Test splitting.


> [!IMPORTANT]  
> Due to external dependencies it requires its own, separate conda environment with its own dependencies.

## Setup Conda 


    conda create -n data python=3.6
    conda activate data

#### Bundle Adjustment Dependency

For the Bundle Adjustment Step the following dependency needs to be installed:
[sat-bundleadjust](https://github.com/centreborelli/sat-bundleadjust). 

    git clone https://github.com/centreborelli/sat-bundleadjust.git
    # need to downgrade setuptools to correctly build sat-bundleadjust 
    pip install --upgrade pip setuptools==57.5.0  
    # manually install openv, otherwise sat-bundleadjust manually builds the newest version which fails 
    pip install opencv-contrib-python==4.5.4.60
    pip install -e ./sat-bundleadjust


#### Conda Dependencies

    conda install gdal rasterio


#### Pip Dependencies

    pip install fire toml pydantic xmltodict plyflatten pygdal==3.0.2.10 black pycocotools







## Configuration and Preprocessing

### How to use

    python -m data_prep.create_dataset <path_to_config='configs/data/dataset.cfg'>

If given a path to a non-existent file, a new `.cfg` file with the specified name
will be created based on a template.
Configure it to fit your needs, following the directions specified in the `.cfg`.
If no path is given, the script will use `configs/data/dataset.cfg` as default input.

