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

from distutils.core import setup, Extension
from pybind11.setup_helpers import Pybind11Extension

ext_modules = [
        # Pybind11Extension("umap_extend", ["umap_extend.cpp", "xsrandom.cpp", "counters.c"], \
        Pybind11Extension("umap_extend", ["umap_extend.cpp", "xsrandom.cpp"], \
                        author="Narendra Chaudhary", \
                        author_email="narendra.chaudhary@intel.com", \
                        # extra_compile_args=['-O3','-qopenmp-simd', '-qopenmp', '-march=native', \
                        #                     '-xCOMMON-AVX512', '-qopt-zmm-usage=high', \
                        #                     '-DVTUNE_ANALYSIS', '-I/swtools/intel/vtune_amplifier/include/', '-littnotify', '-L/swtools/intel/vtune_amplifier/lib64/', \
                        #                 ]),
                        extra_compile_args=['-O3','-fopenmp-simd', '-fopenmp', 
                                            '-march=native', \
                                        #     '-mprefer-vector-width=512', '-mavx512f', '-mavx512cd', '-mavx512bw', \
                                        #     '-mavx512dq', '-mavx512vl', '-mavx512ifma', '-mavx512vbmi', \
                                        ]),
]
setup(name="umap_extend",
        version = "1.0",
        description="This is a package for umap extension.",
        ext_modules= ext_modules)