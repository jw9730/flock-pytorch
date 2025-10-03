from glob import glob

import pybind11
from setuptools import Extension, find_packages, setup


ext_modules = [
    Extension(
        '_walker',
        sorted(glob("src/*.cpp")),
        language='c++',
        include_dirs=[
            pybind11.get_include(),
            pybind11.get_include(True),
        ],
        extra_compile_args=["-O3", "-std=c++11"])
]


setup(
    name="graph-walker",
    description="Fast random walks on graphs and text conversion " \
        "based on graph-walker 1.0.6 by Maixent Chenebaux " \
        "https://github.com/kerighan/graph-walker",
    packages=find_packages(),
    include_package_data=True,
    install_requires=["networkx", "numpy", "scipy", "pybind11", "scikit-learn"],
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    ext_modules=ext_modules
)
