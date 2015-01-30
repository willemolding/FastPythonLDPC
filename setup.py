from distutils.core import setup
from Cython.Build import cythonize

setup(
    name='cython_ldpc_decode',
    ext_modules=cythonize('cython_ldpc_decode.pyx'),
)
