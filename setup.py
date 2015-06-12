from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import os
import numpy

gco_directory = "src"

#files = ['block.h', 'energy.h', 'expand.h',
#         'graph.h']
files = []

#files = [os.path.join(gco_directory, f) for f in files]
files.insert(0, "hop_python.pyx")

setup(cmdclass={'build_ext': build_ext},
      ext_modules=[Extension("robustpn", ["hop_python.pyx"], language="c++",
                             include_dirs=[gco_directory, numpy.get_include()],
                             library_dirs=[gco_directory],
                             extra_compile_args=["-fpermissive"])])
