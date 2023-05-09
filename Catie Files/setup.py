from setuptools import setup
from Cython.Build import cythonize
import numpy as np

setup(
    name="CatieAgentC",
    ext_modules=cythonize("CatieAgentC.pyx"),
    include_dirs=[np.get_include()],
)