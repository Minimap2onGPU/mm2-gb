
## Create an Anaconda environment
```bash
conda create --name umap_env python=3.8.0
conda activate umap_env
```

## Load library path for libmkl_rt.so
```bash
export LD_LIBRARY_PATH=~/anaconda3/umap_env/lib/         # If loading from anaconda3
```

## Install scanpy
```bash
pip install scanpy==1.8.1
```

## Install pybind11
```bash
pip install pybind11
```

## Install umap_extend and umap 
```bash

pip uninstall umap-learn
cd umap_extend
python setup.py install             


cd ../umap
python setup.py install
```

## Run Test
```
python test_umap_fast.py
```