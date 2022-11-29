from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext as _build_ext
import os
import re
import codecs
import platform
from distutils.sysconfig import get_config_var, get_python_inc
from distutils.version import LooseVersion
import sys
import subprocess

# to publish use:
# > python setup.py sdist bdist_wheel upload
# which depends on ~/.pypirc


# This is copied from @robbuckley's fix for Panda's
# For mac, ensure extensions are built for macos 10.9 when compiling on a
# 10.9 system or above, overriding distuitls behavior which is to target
# the version that python was built for. This may be overridden by setting
# MACOSX_DEPLOYMENT_TARGET before calling setup.pcuda-comp-generalizey
if sys.platform == 'darwin':
    if 'MACOSX_DEPLOYMENT_TARGET' not in os.environ:
        current_system = LooseVersion(platform.mac_ver()[0])
        python_target = LooseVersion(get_config_var('MACOSX_DEPLOYMENT_TARGET'))
        if python_target < '10.9' and current_system >= '10.9':
            os.environ['MACOSX_DEPLOYMENT_TARGET'] = '10.9'

here = os.path.abspath(os.path.dirname(__file__))


def read(*parts):
    with codecs.open(os.path.join(here, *parts), 'r') as fp:
        return fp.read()


def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


# Extend the default build_ext class to bootstrap numpy installation
# that are needed to build C extensions.
# see https://stackoverflow.com/questions/19919905/how-to-bootstrap-numpy-installation-in-setup-py
class build_ext(_build_ext):
    def finalize_options(self):
        _build_ext.finalize_options(self)
        if isinstance(__builtins__, dict):
            __builtins__["__NUMPY_SETUP__"] = False
        else:
            setattr(__builtins__, "__NUMPY_SETUP__", False)
        import numpy
        print("numpy.get_include()", numpy.get_include())
        self.include_dirs.append(numpy.get_include())


def find_in_path(name, path):
    """Find a file in a search path and return its full path."""
    # adapted from:
    # http://code.activestate.com/recipes/52224-find-a-file-given-a-search-path/
    for dir in path.split(os.pathsep):
        binpath = os.path.join(dir, name)
        if os.path.exists(binpath):
            return os.path.abspath(binpath)
    return None


def run_setup(with_binary, with_openmp, test_xgboost, test_lightgbm, test_catboost, test_spark, test_pyod,
              test_transformers, test_pytorch, test_sentencepiece, test_opencv):
    ext_modules = []
    if with_binary:
        compile_args = []
        link_args = []
        if sys.platform == 'zos':
            compile_args.append('-qlonglong')
        if sys.platform == 'win32':
            compile_args.append('/MD')
        if with_openmp:
            if sys.platform == "darwin":
                compile_args += ['-Xpreprocessor', '-fopenmp', '-lomp']
            else:
                compile_args.append('-fopenmp')
                link_args.append('-fopenmp')

        ext_modules.append(
            Extension('fasttreeshap._cext', sources=['fasttreeshap/cext/_cext.cc'],
                      extra_compile_args=compile_args, extra_link_args=link_args))

    tests_require = ['pytest', 'pytest-mpl', 'pytest-cov']
    if test_xgboost:
        tests_require += ['xgboost']
    if test_lightgbm:
        tests_require += ['lightgbm']
    if test_catboost:
        tests_require += ['catboost']
    if test_spark:
        tests_require += ['pyspark']
    if test_pyod:
        tests_require += ['pyod']
    if test_transformers:
        tests_require += ['transformers']
    if test_pytorch:
        tests_require += ['torch']
    if test_sentencepiece:
        tests_require += ['sentencepiece']
    if test_opencv:
        tests_require += ['opencv-python']

    extras_require = {
        'plots': [
            'matplotlib',
            'ipython'
        ],
        'others': [
            'lime',
        ],
        'docs': [
            'matplotlib',
            'ipython',
            'numpydoc',
            'sphinx_rtd_theme',
            'sphinx',
            'nbsphinx',
        ]
    }
    extras_require['test'] = tests_require
    extras_require['all'] = list(set(i for val in extras_require.values() for i in val))

    setup(
        name='fasttreeshap',
        version=find_version("fasttreeshap", "__init__.py"),
        description='A fast implementation of TreeSHAP algorithm.',
        long_description="FastTreeSHAP package implements two new algorithms FastTreeSHAP v1 and FastTreeSHAP v2, to improve the computational "
                        +"efficiency of TreeSHAP for large datasets. In practice FastTreeSHAP v1 is 1.5x faster than TreeSHAP while keeping "
                        +"the memory cost unchanged, and FastTreeSHAP v2 is 2.5x faster than TreeSHAP at the cost of a slightly higher memory usage.",
        long_description_content_type="text/markdown",
        url='http://github.com/linkedin/fasttreeshap',
        author='Jilei Yang',
        author_email='jlyang0712@gmail.com',
        license='BSD 2-CLAUSE',
        packages=[
            'fasttreeshap', 'fasttreeshap.explainers', 'fasttreeshap.plots', 'fasttreeshap.plots.colors',
            'fasttreeshap.maskers', 'fasttreeshap.utils', 'fasttreeshap.models'
        ],
        package_data={'fasttreeshap': ['plots/resources/*', 'cext/tree_shap.h']},
        cmdclass={'build_ext': build_ext},
        setup_requires=['numpy'],
        install_requires=['numpy', 'scipy', 'scikit-learn', 'pandas', 'tqdm>4.25.0',
                          'packaging>20.9', 'slicer==0.0.7', 'numba', 'cloudpickle', 'psutil', 'shap'],
        extras_require=extras_require,
        ext_modules=ext_modules,
        classifiers=[
            "Operating System :: Microsoft :: Windows",
            "Operating System :: POSIX",
            "Operating System :: Unix",
            "Operating System :: MacOS",
            "Programming Language :: Python :: 3.7",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
            "Programming Language :: Python :: 3.10",
        ],
        zip_safe=False
        # python_requires='>3.0' we will add this at some point
    )


def try_run_setup(**kwargs):
    """ Fails gracefully when various install steps don't work.
    """

    try:
        run_setup(**kwargs)
    except Exception as e:
        print(str(e))
        if "xgboost" in str(e).lower():
            kwargs["test_xgboost"] = False
            print("Couldn't install XGBoost for testing!")
            try_run_setup(**kwargs)
        elif "lightgbm" in str(e).lower():
            kwargs["test_lightgbm"] = False
            print("Couldn't install LightGBM for testing!")
            try_run_setup(**kwargs)
        elif "catboost" in str(e).lower():
            kwargs["test_catboost"] = False
            print("Couldn't install CatBoost for testing!")
            try_run_setup(**kwargs)
        elif kwargs["with_binary"]:
            kwargs["with_binary"] = False
            print("WARNING: The C extension could not be compiled, sklearn tree models not supported.")
            try_run_setup(**kwargs)
        elif "pyod" in str(e).lower():
            kwargs["test_pyod"] = False
            print("Couldn't install PyOD for testing!")
            try_run_setup(**kwargs)
        elif "transformers" in str(e).lower():
            kwargs["test_transformers"] = False
            print("Couldn't install Transformers for testing!")
            try_run_setup(**kwargs)
        elif "torch" in str(e).lower():
            kwargs["test_pytorch"] = False
            print("Couldn't install PyTorch for testing!")
            try_run_setup(**kwargs)
        elif "sentencepiece" in str(e).lower():
            kwargs["test_sentencepiece"] = False
            print("Couldn't install sentencepiece for testing!")
            try_run_setup(**kwargs)
        else:
            print("ERROR: Failed to build!")


# we seem to need this import guard for appveyor
if __name__ == "__main__":
    try_run_setup(
        with_binary=True, with_openmp=True, test_xgboost=True, test_lightgbm=True, test_catboost=True,
        test_spark=True, test_pyod=True, test_transformers=True, test_pytorch=True,
        test_sentencepiece=True, test_opencv=True
    )
