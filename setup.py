import os
import numpy as np
from setuptools import setup
from Cython.Build import cythonize

PACKAGE_DIR = "lm_fit"

setup(
    name="nonlinear fit",
    version="0.0.1",
    description="Nonlinear fit using Levenberg-Marquardt algorithm",
    author="Dong Nai-ping",
    author_email="nai-ping.dong@polyu.edu.hk",
    packages=[
        "lm_fit",
    ],
    ext_modules=cythonize([
        os.path.join(PACKAGE_DIR, "*.pyx")
    ],
        compiler_directives={'language_level': "3"}
    ),
    include_dirs=[
        np.get_include()
    ]
)
