"""
MIT License

Copyright (c) 2022 Intel Labs

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Authors: Narendra Chaudhary <narendra.chaudhary@intel.com>; Sanchit Misra <sanchit.misra@intel.com>
"""



"""
This example needs scanpy package to run. Please install using pip install scanpy==1.8.1 and uninstall the existing umap-learn package.
"""

import numpy as np
import mkl
import scanpy as sc
import os
import time

# UMAP
umap_min_dist = 0.3 
umap_spread = 1.0

sc.settings.n_jobs = 56            # Set it to number of cpus on a CPU socket

os.environ["OMP_NUM_THREADS"] = str(sc.settings.n_jobs)
mkl.set_num_threads(sc.settings.n_jobs)

adata = sc.read('before_umap.h5ad')
print(adata.shape)
umap_time = time.time()
sc.tl.umap(adata, min_dist=umap_min_dist, spread=umap_spread)
print("UMAP time : %s" % (time.time() - umap_time))

sc.pl.umap(adata, color=["Stmn2_raw"], color_map="Blues", vmax=1, vmin=-0.05, save="_Stmn2_raw.png")
sc.pl.umap(adata, color=["Hes1_raw"], color_map="Blues", vmax=1, vmin=-0.05, save="_Hes1_raw.png")

